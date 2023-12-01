from datasets import load_dataset
import torch
import torch.nn as nn
from transformers import BertTokenizerFast, BertForTokenClassification, BertConfig
from torch.utils.data import DataLoader, Dataset
from seqeval.metrics import classification_report as seqeval_classification_report
import numpy as np
from tqdm.auto import tqdm


class SlotFillingDataset:
    def __init__(
        self,
        tokenizer=BertTokenizerFast.from_pretrained("bert-base-uncased"),
        max_length=128,
    ):
        """
        Initializes the SlotFillingDataset class.
        :param tokenizer: Tokenizer object used for tokenizing texts.
        :param max_length: Maximum length of the tokenized inputs.
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def load(self, dataset_name="multi_woz_v22"):
        """
        Loads the dataset.
        :param dataset_name: Name of the dataset to load.
        """
        try:
            dataset = load_dataset(dataset_name)
            self.train_dataset = dataset["train"]
            self.val_dataset = dataset["validation"]
            self.test_dataset = dataset["test"]
        except Exception as e:
            print(f"Error loading dataset: {e}")

    def get_relevant_data(self, services={"restaurant", "hotel"}):
        """
        Filters the dataset to include only entries with specified services.
        :param services: A set of services to filter the data.
        """
        if not self.train_dataset or not self.val_dataset or not self.test_dataset:
            print("Dataset not loaded.")
            return
        # Filter the dataset
        self.train_dataset = [
            entry
            for entry in self.train_dataset
            if set(entry["services"]).issubset(services)
        ]
        self.val_dataset = [
            entry
            for entry in self.val_dataset
            if set(entry["services"]).issubset(services)
        ]
        self.test_dataset = [
            entry
            for entry in self.test_dataset
            if set(entry["services"]).issubset(services)
        ]

    def create_labelled_data(self, dataset):
        """
        Creates the labelled data.
        :param dataset: Dataset to create the labelled data from.
        """

        # Initialize the labelled data
        labelled_data = []

        for dialogue in dataset:
            turns = dialogue["turns"]
            for i, _ in enumerate(turns["turn_id"]):
                utterance = turns["utterance"][i]

                # Tokenize the combined text (do not use history for labelling)
                encoded = self.tokenizer(
                    utterance,
                    add_special_tokens=True,
                    return_offsets_mapping=True,
                    padding="max_length",
                    truncation=True,
                    max_length=self.max_length,
                )
                tokens = self.tokenizer.convert_ids_to_tokens(encoded["input_ids"])
                attention_mask = encoded["attention_mask"]
                offset_mapping = encoded["offset_mapping"]

                # Initialize labels
                labels = ["O"] * len(tokens)

                # If the turn has dialogue acts, label the tokens
                if "dialogue_acts" in turns and i < len(turns["dialogue_acts"]):
                    act = turns["dialogue_acts"][i]
                    span_info = act.get("span_info", {})

                    # Iterate over the span info
                    for act_slot_name, act_slot_value, span_start, span_end in zip(
                        span_info.get("act_slot_name", []),
                        span_info.get("act_slot_value", []),
                        span_info.get("span_start", []),
                        span_info.get("span_end", []),
                    ):
                        # Initialize the start and end token index
                        start_token_idx = None
                        end_token_idx = None

                        # Find the tokens corresponding to the span
                        for idx, offset in enumerate(offset_mapping):
                            if start_token_idx is None and offset[0] == span_start:
                                start_token_idx = idx
                            if offset[1] == span_end:
                                end_token_idx = idx
                                break

                        # If the span is found, label the tokens
                        if start_token_idx is not None and end_token_idx is not None:
                            if start_token_idx < len(tokens) and end_token_idx < len(
                                tokens
                            ):
                                # Label tokens using IOB format with the actual ground truth slot value
                                labels[start_token_idx] = f"B-{act_slot_name}"
                                for j in range(start_token_idx + 1, end_token_idx + 1):
                                    labels[j] = f"I-{act_slot_name}"
                            else:
                                print(
                                    f"Warning: Index out of range for utterance '{utterance}' with span {span_start}-{span_end}"
                                )
                # Add the encoded text and labels to the labelled data
                labelled_data.append((encoded, labels))

        return labelled_data

    def create_labelled_dialogue_data(self, dataset):
        """
        Creates the labelled data in the format (utterance, {slots, values}) for each dialogue.
        :param dataset: Dataset to create the labelled data from.
        """
        labelled_data = []

        for dialogue in dataset:
            turns = dialogue["turns"]
            dialogue_data = []
            for i, _ in enumerate(turns["turn_id"]):
                utterance = turns["utterance"][i]
                slot_values = {}

                # If the turn has dialogue acts, extract slots and values
                if "dialogue_acts" in turns and i < len(turns["dialogue_acts"]):
                    act = turns["dialogue_acts"][i]
                    span_info = act.get("span_info", {})

                    for act_slot_name, act_slot_value in zip(
                        span_info.get("act_slot_name", []),
                        span_info.get("act_slot_value", []),
                    ):
                        slot_values[act_slot_name] = act_slot_value

                # Append the utterance and extracted slot values to the dialogue data
                dialogue_data.append((utterance, slot_values))

            # Add each complete dialogue to the labelled data
            labelled_data.append(dialogue_data)

        return labelled_data

    def create_label2id(self, labelled_data):
        """
        Creates the label2id mapping.
        :param labelled_data: Processed data from create_labelled_data method (dictionary).
        Returns:
        label2id: Mapping from labels to ids.
        num_labels: Number of unique labels.
        """
        unique_labels = set()
        for _, labels in labelled_data:
            unique_labels.update(set(labels))
        label2id = {label: idx for idx, label in enumerate(sorted(unique_labels))}
        return label2id, len(unique_labels)

    def id2label(self, label2id):
        """
        Creates the id2label mapping.
        :param label2id: Mapping from labels to ids.
        """
        id2label = {idx: label for label, idx in label2id.items()}
        return id2label


class TensorDataset(Dataset):
    def __init__(self, labelled_data, label2id):
        """
        Initializes the SlotFillingData class.
        :param labelled_data: Processed data from create_labelled_data method [(encoded, label)].
        :param label2id: Mapping from labels to ids.
        """
        self.labelled_data = labelled_data
        self.label2id = label2id
        self.attention_mask = None
        self.input_ids = None
        self.labels = None

    def create_tensors(self):
        """
        Creates the tensors.
        """
        self.input_ids = torch.tensor([t[0]["input_ids"] for t in self.labelled_data])
        self.attention_mask = torch.tensor(
            [t[0]["attention_mask"] for t in self.labelled_data]
        )
        self.labels = [
            [self.label2id[label] for label in t[1]] for t in self.labelled_data
        ]
        self.labels = torch.tensor(self.labels)

    def create_dataloader(self, batch_size=32):
        """
        Creates the dataloader.
        :param batch_size: Batch size for training.
        """
        dataset = torch.utils.data.TensorDataset(
            self.input_ids, self.attention_mask, self.labels
        )
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)
        return dataloader


class SlotFillingModel:
    def __init__(
        self,
        train_dataloader,
        val_dataloader,
        test_dataloader,
        tokenizer,
        num_labels,
        label2id,
        id2label,
        max_length=128,
        batch_size=32,
        epochs=3,
        patience=2,
        lr=2e-5,
    ):
        """
        Initializes the SlotFilling class.
        :param dataset: SlotFillingDataset object.
        :param model: Model to use for training.
        :param tokenizer: Tokenizer object used for tokenizing texts.
        :param max_length: Maximum length of the tokenized inputs.
        :param batch_size: Batch size for training.
        :param epochs: Number of epochs for training.
        :param lr: Learning rate for training.
        :param warmup_steps: Warmup steps for training.
        :param patience: Patience for early stopping.
        :param num_labels: Number of labels.
        :param label2id: Mapping from labels to ids.
        :param id2label: Mapping from ids to labels.
        """
        self.dataset = dataset
        self.model = None
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.batch_size = batch_size
        self.epochs = epochs
        self.patience = patience
        self.lr = lr
        self.num_labels = num_labels
        self.label2id = label2id
        self.id2label = id2label
        self.device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.test_dataloader = test_dataloader

    def create_model(self, model_name="bert-base-uncased"):
        """
        Creates the model.
        :param model_name: Name of the model to use, default is 'bert-base-uncased'.
        """
        self.model = BertForTokenClassification.from_pretrained(
            model_name, num_labels=self.num_labels
        ).to(self.device)

    def train(self):
        if not self.model:
            print("Model not created.")
            return

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        criterion = nn.CrossEntropyLoss(ignore_index=-100)

        # Initialize the early stopping counter
        best_val_loss = float("inf")
        patience_counter = 0

        # Training loop
        for epoch in range(self.epochs):
            self.model.train()
            train_loss = 0
            train_progress_bar = tqdm(
                self.train_dataloader,
                desc=f"Epoch {epoch+1}/{self.epochs} Training",
                leave=False,
            )

            # Training phase
            for batch in train_progress_bar:
                batch = tuple(t.to(self.device) for t in batch)
                b_input_ids, b_input_mask, b_labels = batch
                optimizer.zero_grad()
                outputs = self.model(
                    b_input_ids,
                    token_type_ids=None,
                    attention_mask=b_input_mask,
                    labels=b_labels,
                )
                loss = outputs.loss
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
                train_progress_bar.set_postfix(train_loss=loss.item())

            avg_train_loss = train_loss / len(self.train_dataloader)
            print(f"Epoch {epoch + 1}/{self.epochs} | Train Loss: {avg_train_loss}")

            # Validation phase
            self.model.eval()
            val_loss = 0
            val_progress_bar = tqdm(
                self.val_dataloader,
                desc=f"Epoch {epoch+1}/{self.epochs} Validation",
                leave=False,
            )
            for batch in val_progress_bar:
                batch = tuple(t.to(self.device) for t in batch)
                b_input_ids, b_input_mask, b_labels = batch
                with torch.no_grad():
                    outputs = self.model(
                        b_input_ids,
                        token_type_ids=None,
                        attention_mask=b_input_mask,
                        labels=b_labels,
                    )
                    loss = outputs.loss
                    val_loss += loss.item()
                    val_progress_bar.set_postfix(val_loss=loss.item())

            avg_val_loss = val_loss / len(self.val_dataloader)
            print(f"Epoch {epoch + 1}/{self.epochs} | Validation Loss: {avg_val_loss}")

            # Check if the validation loss is lower than the best one seen so far
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                torch.save(self.model.state_dict(), f"checkpoint_epoch_{epoch+1}.pt")
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    print("Early stopping!")
                    break
        print("Training complete. Final model saved.")

    def test(self):
        if not self.model:
            print("Model not created.")
            return

        self.model.eval()
        total_loss = 0
        all_predictions = []
        all_true_labels = []

        with torch.no_grad():
            for batch in self.test_dataloader:
                batch = tuple(t.to(self.device) for t in batch)
                b_input_ids, b_attention_masks, b_labels = batch

                outputs = self.model(
                    b_input_ids,
                    token_type_ids=None,
                    attention_mask=b_attention_masks,
                    labels=b_labels,
                )
                loss = outputs.loss
                total_loss += loss.item()

                # Move logits and labels to CPU
                logits = outputs.logits.detach().cpu().numpy()
                label_ids = b_labels.to("cpu").numpy()

                # Convert logits to token predictions
                predictions = np.argmax(logits, axis=-1)

                # For each item in the batch...
                for i in range(b_input_ids.size(0)):
                    # Skip predictions for tokens with label_id == -100
                    pred_label_sequence = []
                    true_label_sequence = []
                    for j, (pred_id, label_id) in enumerate(
                        zip(predictions[i], label_ids[i])
                    ):
                        if b_attention_masks[i][j] != 0 and label_id != -100:
                            pred_label_sequence.append(
                                self.id2label.get(pred_id, "O")
                            )  # Default to 'O' if key is not found
                            true_label_sequence.append(self.id2label[label_id])

                    # Ensure the true and predicted sequences have the same length
                    if len(true_label_sequence) != len(pred_label_sequence):
                        print(
                            f"Length mismatch in sequence {i}: true labels {len(true_label_sequence)} vs. predicted labels {len(pred_label_sequence)}"
                        )
                        # Output the actual sequences to help diagnose the issue
                        print("True labels:", true_label_sequence)
                        print("Pred labels:", pred_label_sequence)
                        continue

                    # ...extend the true labels and predicted labels lists
                    all_true_labels.append(true_label_sequence)
                    all_predictions.append(pred_label_sequence)

        # Calculate average loss over all the batches
        avg_loss = total_loss / len(self.test_dataloader)
        print(f"Test loss: {avg_loss}")

        # Use seqeval to compute a classification report
        seqeval_report = seqeval_classification_report(all_true_labels, all_predictions)
        print(seqeval_report)

    def query(self, utterance):
        if not self.model:
            print("Model not created.")
            return

        self.model.eval()
        with torch.no_grad():
            encoded_input = self.tokenizer(
                utterance,
                add_special_tokens=True,
                return_offsets_mapping=True,
                padding="max_length",
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )

        # Move tensors to the correct device
        input_ids = encoded_input["input_ids"].to(self.device)
        attention_masks = encoded_input["attention_mask"].to(self.device)

        # Forward pass
        with torch.no_grad():
            outputs = self.model(input_ids, attention_mask=attention_masks)
            logits = outputs.logits
            logits = logits.detach().cpu()

        # Use logits and id2label to get the predicted labels
        predictions = torch.argmax(logits, dim=2).squeeze().tolist()
        offset_mapping = encoded_input["offset_mapping"].squeeze().tolist()

        # Map predictions back to original words
        word_labels = []
        last_word_end = None
        for label_id, offset in zip(predictions, offset_mapping):
            word_start, word_end = offset

            # Check if this is the start of a new word
            if word_start != last_word_end:
                word_label = self.id2label.get(label_id, "O")
                word_labels.append(word_label)

            last_word_end = word_end

        return word_labels
    
    def query_slots(self, utterance):
        if not self.model:
            print("Model not created.")
            return

        self.model.eval()
        with torch.no_grad():
            encoded_input = self.tokenizer(
                utterance,
                add_special_tokens=True,
                return_offsets_mapping=True,
                padding="max_length",
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )

        # Move tensors to the correct device
        input_ids = encoded_input["input_ids"].to(self.device)
        attention_masks = encoded_input["attention_mask"].to(self.device)

        # Forward pass
        with torch.no_grad():
            outputs = self.model(input_ids, attention_mask=attention_masks)
            logits = outputs.logits
            logits = logits.detach().cpu()

        # Use logits and id2label to get the predicted labels
        predictions = torch.argmax(logits, dim=2).squeeze().tolist()
        offset_mapping = encoded_input["offset_mapping"].squeeze().tolist()
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids.squeeze().tolist())

        # Extracting slot values from the utterance
        slot_values = {}
        current_slot = None
        current_value = ""

        for token, label_id, (word_start, word_end) in zip(tokens, predictions, offset_mapping):
            label = self.id2label.get(label_id, "O")

            if label.startswith("B-"):
                # Save the previous slot and value if any
                if current_slot:
                    slot_values[current_slot] = current_value.strip()
                
                current_slot = label[2:]  # Remove the 'B-' prefix
                current_value = utterance[word_start:word_end]

            elif label.startswith("I-") and current_slot:
                current_value += " " + utterance[word_start:word_end]

            elif label == "O":
                # Save the previous slot and value if any
                if current_slot:
                    slot_values[current_slot] = current_value.strip()
                    current_slot = None
                    current_value = ""

        # Save the last found slot and value
        if current_slot:
            slot_values[current_slot] = current_value.strip()

        return slot_values

    

def train_and_test_model():
    # Create the dataset
    dataset = SlotFillingDataset()
    # Load the dataset
    dataset.load()
    # Filter the dataset
    dataset.get_relevant_data({"restaurant", "hotel"})
    # Create the labelled training data
    labelled_data = dataset.create_labelled_data(dataset.train_dataset)
    # Create the labelled validation data
    labelled_data_val = dataset.create_labelled_data(dataset.val_dataset)
    # Create the labelled test data
    labelled_data_test = dataset.create_labelled_data(dataset.test_dataset)
    # Create the label2id mapping
    label2id, num_labels = dataset.create_label2id(labelled_data)
    # Create the TensorDataset for training
    tensor_dataset_train = TensorDataset(labelled_data, label2id)
    tensor_dataset_train.create_tensors()
    # Create the TensorDataset for validation
    tensor_dataset_val = TensorDataset(labelled_data_val, label2id)
    tensor_dataset_val.create_tensors()
    # Create the TensorDataset for test
    tensor_dataset_test = TensorDataset(labelled_data_test, label2id)
    tensor_dataset_test.create_tensors()
    # Create the dataloader for training
    train_dataloader = tensor_dataset_train.create_dataloader()
    # Create the dataloader for validation
    val_dataloader = tensor_dataset_val.create_dataloader()
    # Create the dataloader for test
    test_dataloader = tensor_dataset_test.create_dataloader()
    # Create the model
    model = SlotFillingModel(
        train_dataloader,
        val_dataloader,
        test_dataloader,
        dataset.tokenizer,
        num_labels,
        label2id,
        dataset.id2label(label2id),
    )
    model.create_model()
    # Train the model
    model.train()
    # Test the model
    model.test()
    # Query the model
    res = model.query("I want to book a table for 4 people at 7pm tonight.")
    print(res)

if __name__ == "__main__":
    
    # Create dialogues dataset'
    dataset = SlotFillingDataset()
    # Load the dataset
    dataset.load()
    # Filter the dataset
    dataset.get_relevant_data({"restaurant", "hotel"})
    # Create the labelled training data
    labelled_data = dataset.create_labelled_data(dataset.train_dataset)
    # Create the labelled validation data
    labelled_data_val = dataset.create_labelled_data(dataset.val_dataset)
    # Create the labelled test data
    labelled_data_test = dataset.create_labelled_data(dataset.test_dataset)
    # Create the label2id mapping
    label2id, num_labels = dataset.create_label2id(labelled_data)
    # Create the TensorDataset for training
    tensor_dataset_train = TensorDataset(labelled_data, label2id)
    tensor_dataset_train.create_tensors()
    # Create the TensorDataset for validation
    tensor_dataset_val = TensorDataset(labelled_data_val, label2id)
    tensor_dataset_val.create_tensors()
    # Create the TensorDataset for test
    tensor_dataset_test = TensorDataset(labelled_data_test, label2id)
    tensor_dataset_test.create_tensors()
    # Create the dataloader for training
    train_dataloader = tensor_dataset_train.create_dataloader()
    # Create the dataloader for validation
    val_dataloader = tensor_dataset_val.create_dataloader()
    # Create the dataloader for test
    test_dataloader = tensor_dataset_test.create_dataloader()
    # Create the model
    dialogue_data = dataset.create_labelled_dialogue_data(dataset.test_dataset)
    # create model from checkpoint
    model = SlotFillingModel(
    train_dataloader,
    val_dataloader,
    test_dataloader,
    dataset.tokenizer,
    num_labels,
    label2id,
    dataset.id2label(label2id),
)
    model.create_model()
    model.model.load_state_dict(torch.load("checkpoint_epoch_3.pt"))
    model.model.eval()
    # Query the model
    res = model.query_slots("I want to book a table for 4 people at 7pm tonight.")
    print(res)
    
