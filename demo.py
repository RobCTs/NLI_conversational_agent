# Purpose: Demo file for the project

###################################
from datasets import load_dataset
import torch
import torch.nn as nn
from transformers import (
    BertTokenizerFast,
    BertForTokenClassification,
    DistilBertTokenizer,
    DistilBertModel,
    BertTokenizer,
    BertModel,
)
from sklearn.preprocessing import MultiLabelBinarizer

from torch.utils.data import Dataset
from collections import Counter

from sklearn.metrics import classification_report
from seqeval.metrics import classification_report as seqeval_classification_report
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import typing
import random
import pickle
import joblib
from pydantic import BaseModel

###################################


###########################
# DIALOGUE ACT PREDICTION #
###########################


class DAClassifier(nn.Module):
    def __init__(self, num_classes, bert_model_type="distilbert-base-uncased"):
        super(DAClassifier, self).__init__()
        self.bert = DistilBertModel.from_pretrained(bert_model_type)
        self.fc = nn.Linear(768, num_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[
            :, 0, :
        ]  # selects the [CLS] token position.
        logits = torch.sigmoid(self.fc(cls_output))
        return logits


class DialogItemIdentifier(BaseModel):
    id_dialog: str
    order_in_dialog: int


class DialogItem(BaseModel):
    id_dialog: str
    order_in_dialog: int
    utterance: str
    speaker: str
    dialogue_acts: typing.List[str]
    gt_dialogue_acts: typing.List[str]
    previous_dialog_items: typing.List[DialogItemIdentifier] = []


class DialogActModel:
    PAST_HISTORY_LENGTH = 2

    def __init__(
        self, model_path="", mlb_path="", bert_model_type="distilbert-base-uncased"
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = DistilBertTokenizer.from_pretrained(bert_model_type)
        self.mlb: MultiLabelBinarizer = pickle.load(open(mlb_path, "rb"))
        # Load the best model
        saved_model = DAClassifier(num_classes=len(self.mlb.classes_)).to(self.device)
        best_model = torch.load(model_path, map_location=torch.device(self.device))
        saved_model.load_state_dict(best_model["model_state_dict"])

        self.model = saved_model

    def convert_dialogitem_encoded_history(
        self,
        dialog_item: DialogItem,
        dialog_item_dataset: typing.List[DialogItem],
        past_history_length: int = 2,
    ):
        """
        Converts a DialogItem object into an Encoded History string.

        Parameters:
        - dialog_item: DialogItem object.
        - dialog_item_dataset: List of DialogItem objects.
        - past_history_length: Length of the past history to consider.


        Returns:
        - Encoded History string.
        """
        encoded_history = ""

        # Initialize the history of the user and agent as lists with empty DialogItem objects
        agent_history: typing.List[DialogItem] = []
        user_history: typing.List[DialogItem] = []

        # For each identifier of the previous dialog items, search in the dialog_item_dataset for the corresponding DialogItem object
        # and append it to the agent or user history. If no DialogItem object is found, append a dialog item with empty utterance "" and dialogue acts []
        for i in range(len(dialog_item.previous_dialog_items)):
            for j in range(len(dialog_item_dataset)):
                if (
                    dialog_item_dataset[j].id_dialog
                    == dialog_item.previous_dialog_items[i].id_dialog
                    and dialog_item_dataset[j].order_in_dialog
                    == dialog_item.previous_dialog_items[i].order_in_dialog
                ):
                    if dialog_item_dataset[j].speaker == "Agent":
                        agent_history.append(dialog_item_dataset[j])
                        # If the agent history is longer than the past history length, remove the oldest turn
                        if len(agent_history) > past_history_length:
                            agent_history.pop(0)
                    elif dialog_item_dataset[j].speaker == "User":
                        user_history.append(dialog_item_dataset[j])
                        # If the user history is longer than the past history length, remove the oldest turn
                        if len(user_history) > past_history_length:
                            user_history.pop(0)

        # Fill up the encoded history of the user with the beggining of the array with DialogItems with empty utterance "" and dialogue acts [], for the amount
        # of turns that are missing to reach the past history length
        past_encoded_user_history = ""
        for i in range(past_history_length - len(user_history)):
            past_encoded_user_history += ">".join(["", ""]) + "|"
        for i in range(len(user_history)):
            past_encoded_user_history += (
                ">".join(
                    [user_history[i].utterance, "_".join(user_history[i].dialogue_acts)]
                )
                + "|"
            )

        # Fill up the encoded history of the agent with the beggining of the array with DialogItems with empty utterance "" and dialogue acts [], for the amount
        # of turns that are missing to reach the past history length
        past_encoded_agent_history = ""
        for i in range(past_history_length - len(agent_history)):
            past_encoded_agent_history += ">".join(["", ""]) + "|"

        for i in range(len(agent_history)):
            past_encoded_agent_history += (
                ">".join(
                    [
                        agent_history[i].utterance,
                        "_".join(agent_history[i].dialogue_acts),
                    ]
                )
                + "|"
            )

        encoded_history = (
            past_encoded_user_history
            + past_encoded_agent_history
            + dialog_item.utterance
        )

        return encoded_history

    def predict(self, encoded_history):
        """
        Given an encoded history, predicts the dialogue acts of the last turn.
        """
        self.model.eval()

        # Separate the history from the current utterance splitting by the last "|" character, but don't remove it
        history, utterance = encoded_history.rsplit("|", 1)

        # Merge the history and sentence into a single string adding a [SEP] token between them
        encoded_history = "".join(history) + " [SEP] " + utterance

        encoded = self.tokenizer.encode_plus(
            encoded_history,
            add_special_tokens=True,
            max_length=256,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
        )

        input_ids = torch.tensor([encoded["input_ids"]], dtype=torch.long).to(
            self.device
        )
        attention_mask = torch.tensor([encoded["attention_mask"]], dtype=torch.long).to(
            self.device
        )

        # Make a prediction
        with torch.no_grad():
            logits = self.model(input_ids, attention_mask)

        logits_cpu = logits.to("cpu")
        return logits_cpu.numpy()

    def predict_only_last_dialog_item(
        self, dialog_item_dataset: typing.List[DialogItem]
    ) -> typing.List[str]:
        """
        Predicts the dialogue act of last User turn in the dialog. For the Agent, the DAs are filled using the ground truth from the pre-processing function and it arrives here filled, since
        we know what dialog acts the agent is performing. For previous User DAs, the ground truth is also used (allowed by professors)

        Parameters:
        - dialog_item_dataset: List of DialogItem objects.


        Returns:
        - List of predicted dialogue acts
        """

        dataset_copy = dialog_item_dataset.copy()

        # If the speaker is the User, predict the dialogue act
        encoded_history = self.convert_dialogitem_encoded_history(
            dataset_copy[-1], dataset_copy
        )
        preds = self.predict(encoded_history)
        threshold = 0.5
        all_preds_binary = []
        for all_pred in preds:
            local_pred = []
            for old_local_pred in all_pred:
                binary_local_pred = (old_local_pred > threshold).astype(int)
                local_pred.append(binary_local_pred)
            all_preds_binary.append(local_pred)
        labels_preds = self.mlb.inverse_transform(np.array(all_preds_binary))
        dialogue_acts = labels_preds[0]

        return dialogue_acts

    @staticmethod
    def relabel_dialogue_act(dialogue_act: str):
        if dialogue_act.split("-")[0].upper() not in [
            "RESTAURANT",
            "HOTEL",
            "BOOKING",
            "GENERAL",
        ]:
            new_dialog_act = dialogue_act.split("-")[0]
        else:
            new_dialog_act = dialogue_act

        return new_dialog_act

    @staticmethod
    def add_dialogue_items_to_dialogue_history(
        utterance: str,
        speaker: str,
        dialog_acts: typing.List[str],
        id_dialog: int,
        order_in_dialog: int,
        previous_dialog_history_ids: typing.List[DialogItemIdentifier],
        dialog_history: typing.List[DialogItem],
    ):
        dialogue_act_relabeled = []
        for j in range(len(dialog_acts)):
            dialogue_act_relabeled.append(
                DialogActModel.relabel_dialogue_act(dialog_acts[j])
            )

        # Create a DialogItem object for this turn
        dialog_item = DialogItem(
            id_dialog=id_dialog,
            order_in_dialog=order_in_dialog,
            utterance=utterance,
            speaker=speaker,
            dialogue_acts=dialogue_act_relabeled,
            gt_dialogue_acts=[],
            previous_dialog_items=previous_dialog_history_ids,
        )

        # Append the DialogItem object to the list of DialogItem objects
        dialog_history.append(dialog_item)

        # Append the DialogItemIdentifier object to the history of the user and agent
        dialog_item_identifier = DialogItemIdentifier(
            id_dialog=id_dialog, order_in_dialog=order_in_dialog
        )
        previous_dialog_history_ids.append(dialog_item_identifier)

        # If the history of the user and agent is longer than the past history length multiplied by two, which guarantees that this wont fail on the conversion and speeds up the process
        # , remove the oldest turn
        if len(previous_dialog_history_ids) > DialogActModel.PAST_HISTORY_LENGTH * 2:
            previous_dialog_history_ids.pop(0)

    ################
    # SLOT FILLING #
    ################


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
                    for act_type, act_slot_name, _, span_start, span_end in zip(
                        span_info.get("act_type", []),
                        span_info.get("act_slot_name", []),
                        span_info.get("act_slot_value", []),
                        span_info.get("span_start", []),
                        span_info.get("span_end", []),
                    ):
                        # Initialize the start and end token index
                        start_token_idx = None
                        end_token_idx = None

                        prefix = act_type.split("-")[0].lower()

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
                                labels[start_token_idx] = f"B-{prefix}-{act_slot_name}"
                                for j in range(start_token_idx + 1, end_token_idx + 1):
                                    labels[j] = f"I-{prefix}-{act_slot_name}"
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
        epochs=7,
        patience=2,
        lr=2e-5,
    ):
        """
        Initializes the SlotFillingModel class.
        :param train_dataloader: Dataloader for training.
        :param val_dataloader: Dataloader for validation.
        :param test_dataloader: Dataloader for testing.
        :param tokenizer: Tokenizer object used for tokenizing texts.
        :param num_labels: Number of unique labels.
        :param label2id: Mapping from labels to ids.
        :param id2label: Mapping from ids to labels.
        :param max_length: Maximum length of the tokenized inputs.
        :param batch_size: Batch size for training.
        :param epochs: Number of epochs for training.
        :param patience: Number of epochs to wait for improvement before early stopping.
        :param lr: Learning rate for the optimizer.
        """
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

        for token, label_id, (word_start, word_end) in zip(
            tokens, predictions, offset_mapping
        ):
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


class DialogSlotMemory:
    slot_list_dict: typing.Dict[str, typing.List[str]] = {}

    def __init__(self):
        self.slot_list_dict = {}

    def add_slot(self, slot_name: str, slot_value: str):
        if slot_name not in self.slot_list_dict:
            self.slot_list_dict[slot_name] = []
        self.slot_list_dict[slot_name].append(slot_value)

    def get_slot_values(self, slot_name: str):
        return self.slot_list_dict[slot_name]

    def get_most_recent_slot_value(self, slot_name: str):
        return (
            self.slot_list_dict[slot_name][-1]
            if slot_name in self.slot_list_dict
            else None
        )

    def get_all_slot_values(self):
        return self.slot_list_dict

    def get_all_slot_names(self):
        return self.slot_list_dict.keys()


class SlotValueParser:
    def extract_slot_spans(self, dataset):
        """
        Extracts unique tuples of actual slot values and their spans in the utterance.
        :param dataset: Dataset to extract slot spans from.
        """
        extracted_data = []

        for dialogue in dataset:
            turns = dialogue["turns"]
            for i, _ in enumerate(turns["turn_id"]):
                utterance = turns["utterance"][i]

                # Extracting slot values and their spans
                if "dialogue_acts" in turns and i < len(turns["dialogue_acts"]):
                    act = turns["dialogue_acts"][i]
                    span_info = act.get("span_info", {})

                    for act_slot_name, act_slot_value, span_start, span_end in zip(
                        span_info.get("act_slot_name", []),
                        span_info.get("act_slot_value", []),
                        span_info.get("span_start", []),
                        span_info.get("span_end", []),
                    ):
                        # Finding the actual span in the utterance
                        actual_span = utterance[span_start:span_end]

                        # Storing unique slot value and span tuples
                        slot_span_tuple = (act_slot_name, act_slot_value, actual_span)
                        if slot_span_tuple not in extracted_data:
                            extracted_data.append(slot_span_tuple)

        return extracted_data

    def get_mismatched_slot_values(self, extracted_data):
        """
        Get the mismatched slot values stored as a dataframe
        :param extracted_data: Extracted slot spans from the dataset in the format (slot_name, slot_value, actual_span).
        """

        mismatched_slot_values = []
        for slot_name, slot_value, actual_span in extracted_data:
            if slot_value != actual_span:
                mismatched_slot_values.append([slot_name, slot_value, actual_span])
        mismatched_slot_values = pd.DataFrame(
            mismatched_slot_values, columns=["slot_name", "slot_value", "actual_span"]
        )
        return mismatched_slot_values

    def get_mismatched_slot_values(self, extracted_data):
        """
        Get the mismatched slot values stored as a dataframe
        :param extracted_data: Extracted slot spans from the dataset in the format (slot_name, slot_value, actual_span).
        """

        mismatched_slot_values = []
        for slot_name, slot_value, actual_span in extracted_data:
            if slot_value != actual_span:
                mismatched_slot_values.append([slot_name, slot_value, actual_span])
        mismatched_slot_values = pd.DataFrame(
            mismatched_slot_values, columns=["slot_name", "slot_value", "actual_span"]
        )
        return mismatched_slot_values

    def convert_span_to_slot_value(
        self, slot_name, actual_span, dialogue_slot_memory=None
    ):
        """
        Convert the actual span to the slot value
        :param slot_name: Name of the slot.
        :param actual_span: Actual span in the utterance.
        :param dialogue_slot_memory: Dialogue slot memory object containing the slot values from the dialogue history.
        """

        # Remove the lowercase and whitespace initially
        standardized_span = actual_span.lower().strip()

        # dictionary to convert strings to number representations
        number_dict = {
            "zero": "0",
            "one": "1",
            "two": "2",
            "three": "3",
            "four": "4",
            "five": "5",
            "six": "6",
            "seven": "7",
            "eight": "8",
            "nine": "9",
            "ten": "10",
            "eleven": "11",
            "twelve": "12",
            "thirteen": "13",
            "fourteen": "14",
            "fifteen": "15",
            "sixteen": "16",
            "seventeen": "17",
            "eighteen": "18",
            "nineteen": "19",
            "twenty": "20",
        }

        # if span is a number word, convert it to a number and return
        if standardized_span in number_dict:
            return number_dict[standardized_span]

        # if the slot is a postcode, uppercase the letters and return
        if slot_name == "postcode":
            return standardized_span.upper()

        # arbitrary spelling for "centre" in the dataset
        if standardized_span == "center":
            return "centre"

        # if slot is a time, convert the first word to a number and uppercase the second word
        if slot_name == "booktime":
            # check if the span consists of two words
            if len(standardized_span.split()) >= 2:
                # convert the first word to a number
                partial = number_dict.get(
                    standardized_span.split()[0], standardized_span.split()[0]
                )
                # keep the distance between number and second word e.g.'9 pm': '9 PM'
                standardized_span = partial + " " + standardized_span.split()[1]
                return standardized_span
            else:
                return standardized_span

        # if span indicates indifference to a slot value, return "dontcare"
        if standardized_span in [
            "any",
            "anything",
            "anywhere",
            "dont care",
            "dontcare",
            "dontcare",
            "don't care",
            "do not care",
            "don't really care",
            "doesnt matter",
            "doesn't matter",
            "does not matter",
            "doesn't really matter",
            "not really",
            "no preference",
            "no particular",
            "not particular",
            "either one is fine",
            "either is fine",
            "don't have a preference",
            "do not have a preference",
        ]:
            return "dontcare"

        if slot_name == "name":
            s = standardized_span.split()
            if (
                actual_span.islower()
            ):  # or if s[1][0].isupper(): #for single case: name 'The Allenbell':'the Allenbell'
                for i in range(
                    len(s)
                ):  # name 'Nandos': 'nandos', name 'Lensfield Hotel': 'lensfield hotel'
                    s[i] = s[i].capitalize()
                return " ".join(s)

        # each word in address always capitalized regardless of actual_span value
        if slot_name == "address":
            s = standardized_span.split()
            for i in range(len(s)):
                s[i] = s[i].capitalize()
            return " ".join(s)

        # if span contains 'same' return the relevant value from dialogue history
        if "same" in standardized_span.split() and dialogue_slot_memory is not None:
            # get the most recent slot value from the dialogue history
            return dialogue_slot_memory.get_most_recent_slot_value(slot_name)

        return standardized_span


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


def return_model(model_path="checkpoint_epoch_3.pt"):
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
    model.model.load_state_dict(torch.load(model_path))
    model.model.eval()
    return model


def load_and_query_model(query, model_path="checkpoint_epoch_3.pt"):
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
    model.model.load_state_dict(torch.load(model_path))
    model.model.eval()
    # Query the model
    res = model.query_slots(query)
    print(res)


def test_span_to_slot_value_mapping():
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
    # get the slot spans and values
    parser = SlotValueParser()
    # Get the unique slots
    slot_spans = parser.extract_slot_spans(dataset.train_dataset)
    # Get the mismatched slot values
    mismatched_slot_values = parser.get_mismatched_slot_values(
        slot_spans
    )  # get the mismatched slot values

    fix_count = 0
    total_mismatched_slot_values = len(mismatched_slot_values)

    for slot_name, slot_value, actual_span in mismatched_slot_values.values:
        # convert the actual span to the slot value
        fixed_span = parser.convert_span_to_slot_value(slot_name, actual_span)
        if slot_value == fixed_span:
            print(
                f"Corrected slot value for '{slot_name}': '{actual_span}' -> '{fixed_span}'"
            )
            fix_count += 1
    print(f"Fixed {fix_count}/{total_mismatched_slot_values} slot values.")


def test_full_dialogue_system(
    path_to_model_checkpoint="checkpoint_epoch_3.pt", print_output=False
):
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
    model.model.load_state_dict(torch.load(path_to_model_checkpoint))
    model.model.eval()

    # create labeled dialogue data
    labelled_dialogue_data = dataset.create_labelled_dialogue_data(dataset.test_dataset)

    # query the model for each dialogue

    num_slots_correct = 0
    num_slots_total = 0

    num_labels_correct = 0
    num_labels_total = 0

    num_slots_filled_correct = 0
    num_slots_filled_total = 0

    for dialogue in labelled_dialogue_data:
        dialogue_slot_memory = DialogSlotMemory()
        parser = SlotValueParser()
        for utterance, gt_slot_values in dialogue:
            predicted_slots = model.query_slots(utterance)

            for slot_name, predicted_value in predicted_slots.items():
                # Convert the predicted slot value
                predicted_value = parser.convert_span_to_slot_value(
                    slot_name, predicted_value, dialogue_slot_memory
                )

                # Update the predicted slots dictionary with the converted value
                predicted_slots[slot_name] = predicted_value

                # Update dialogue slot memory
                dialogue_slot_memory.add_slot(slot_name, predicted_value)

            # Compare the dictionaries
            num_slots_total += len(gt_slot_values)
            num_slots_filled_total += len(predicted_slots)

            for slot_name, gt_value in gt_slot_values.items():
                if slot_name in predicted_slots:
                    num_slots_correct += 1
                    if predicted_slots[slot_name] == gt_value:
                        num_slots_filled_correct += 1
                    num_labels_total += 1
                    if gt_value == predicted_slots[slot_name]:
                        num_labels_correct += 1

            if print_output:
                print(f"Utterance: '{utterance}'")
                print(f"Predicted Slots: {predicted_slots}")
                print(f"Ground Truth Slots: {gt_slot_values}")
                print("-" * 50)

    # Number of slots that were identified correctly
    print(f"Slot Accuracy: {num_slots_correct/num_slots_total}")
    # Fraction of values that were parsed correctly
    print(f"Value Accuracy: {num_labels_correct/num_labels_total}")
    # Fraction of slots that were filled correctly
    print(f"Slot/Value Accuracy: {num_slots_filled_correct/num_slots_filled_total}")


#########################
# AGENT MOVE PREDICTION #
#########################

# 3.1:


def relabel_dialogue_act(dialogue_act: str):
    if dialogue_act.split("-")[0].upper() not in [
        "RESTAURANT",
        "HOTEL",
        "BOOKING",
        "GENERAL",
    ]:
        new_dialog_act = dialogue_act.split("-")[0]
    else:
        new_dialog_act = dialogue_act

    return new_dialog_act


def add_dialogue_items_to_dialogue_history_for_agent_move(
    utterance: str,
    speaker: str,
    dialog_acts: typing.List[str],
    to_be_retrieved_gt: typing.List[str],
    id_dialog: int,
    order_in_dialog: int,
    previous_dialog_history_ids: typing.List[DialogItemIdentifier],
    dialog_history: typing.List[DialogItem],
):
    dialogue_act_relabeled = []
    for j in range(len(dialog_acts)):
        dialogue_act_relabeled.append(relabel_dialogue_act(dialog_acts[j]))

    # Create a DialogItem object for this turn
    dialog_item = DialogItemForAgentMove(
        id_dialog=id_dialog,
        order_in_dialog=order_in_dialog,
        utterance=utterance,
        speaker=speaker,
        dialogue_acts=dialogue_act_relabeled,
        to_be_retrieved=to_be_retrieved_gt,
        gt_dialogue_acts=[],
        previous_dialog_items=previous_dialog_history_ids,
    )

    # Append the DialogItem object to the list of DialogItem objects
    dialog_history.append(dialog_item)

    # Append the DialogItemIdentifier object to the history of the user and agent
    dialog_item_identifier = DialogItemIdentifier(
        id_dialog=id_dialog, order_in_dialog=order_in_dialog
    )
    previous_dialog_history_ids.append(dialog_item_identifier)

    # If the history of the user and agent is longer than the past history length multiplied by two, which guarantees that this wont fail on the conversion and speeds up the process
    # , remove the oldest turn
    if (
        len(previous_dialog_history_ids)
        > AgentToBeRetrievedModel.PAST_HISTORY_LENGTH * 2
    ):
        previous_dialog_history_ids.pop(0)


class DialogItemForAgentMove(BaseModel):
    id_dialog: str
    order_in_dialog: int
    utterance: str
    speaker: str
    dialogue_acts: typing.List[str]
    to_be_retrieved: typing.List[str]
    previous_dialog_items: typing.List[DialogItemIdentifier] = []


class BertBILSTMToBeRetrievedClassifier(nn.Module):
    lstm_hidden_size = 256
    num_lstm_layers = 2
    bert_model_type = "distilbert-base-uncased"

    def __init__(self, num_classes: int):
        super(BertBILSTMToBeRetrievedClassifier, self).__init__()
        self.bert = DistilBertModel.from_pretrained(
            BertBILSTMToBeRetrievedClassifier.bert_model_type
        )
        self.lstm = nn.LSTM(
            input_size=768,
            hidden_size=BertBILSTMToBeRetrievedClassifier.lstm_hidden_size,
            num_layers=BertBILSTMToBeRetrievedClassifier.num_lstm_layers,
            batch_first=True,
            bidirectional=True,
        )
        # Corrected linear layer to connect to LSTM output
        self.fc = nn.Linear(
            BertBILSTMToBeRetrievedClassifier.lstm_hidden_size * 2, num_classes
        )  # Multiplied by two because of the bidirectional LSTM

    def forward(self, input_ids, attention_mask):
        bert_output = self.bert(input_ids, attention_mask=attention_mask)
        lstm_output, _ = self.lstm(bert_output.last_hidden_state)

        # Weighted pooling
        seq_len = lstm_output.shape[1]
        weights = (
            torch.linspace(1, 2, seq_len).unsqueeze(0).unsqueeze(2).to(input_ids.device)
        )
        weighted_lstm_output = lstm_output * weights
        weighted_avg_pool = torch.mean(weighted_lstm_output, dim=1)

        logits = torch.sigmoid(self.fc(weighted_avg_pool))
        return logits


class AgentToBeRetrievedModel:
    PAST_HISTORY_LENGTH = 1
    bert_model_type = "distilbert-base-uncased"

    def __init__(self, model_path="", mlb_path=""):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = DistilBertTokenizer.from_pretrained(
            AgentToBeRetrievedModel.bert_model_type
        )
        self.mlb: MultiLabelBinarizer = pickle.load(open(mlb_path, "rb"))
        # Load the best model
        saved_model = BertBILSTMToBeRetrievedClassifier(
            num_classes=len(self.mlb.classes_)
        ).to(self.device)
        best_model = torch.load(model_path, map_location=torch.device(self.device))
        saved_model.load_state_dict(best_model["model_state_dict"])

        self.model = saved_model

    def convert_dialogitem_encoded_history(
        self,
        dialog_item: DialogItemForAgentMove,
        dialog_item_dataset: typing.List[DialogItemForAgentMove],
    ):
        """
        Converts a DialogItem object into an Encoded History string.

        Parameters:
        - dialog_item: DialogItem object.
        - dialog_item_dataset: List of DialogItem objects.
        - past_history_length: Length of the past history to consider.


        Returns:
        - Encoded History string.
        """
        encoded_history = ""

        # Initialize the history of the user and agent as lists with empty DialogItem objects
        agent_history: typing.List[DialogItemForAgentMove] = []
        user_history: typing.List[DialogItemForAgentMove] = []

        # For each identifier of the previous dialog items, search in the dialog_item_dataset for the corresponding DialogItem object
        # and append it to the agent or user history. If no DialogItem object is found, append a dialog item with empty utterance "" and dialogue acts []
        for i in range(len(dialog_item.previous_dialog_items)):
            for j in range(len(dialog_item_dataset)):
                if (
                    dialog_item_dataset[j].id_dialog
                    == dialog_item.previous_dialog_items[i].id_dialog
                    and dialog_item_dataset[j].order_in_dialog
                    == dialog_item.previous_dialog_items[i].order_in_dialog
                ):
                    if dialog_item_dataset[j].speaker == "Agent":
                        agent_history.append(dialog_item_dataset[j])
                        # If the agent history is longer than the past history length, remove the oldest turn
                        if (
                            len(agent_history)
                            > AgentToBeRetrievedModel.PAST_HISTORY_LENGTH
                        ):
                            agent_history.pop(0)
                    elif dialog_item_dataset[j].speaker == "User":
                        user_history.append(dialog_item_dataset[j])
                        # If the user history is longer than the past history length, remove the oldest turn
                        if (
                            len(user_history)
                            > AgentToBeRetrievedModel.PAST_HISTORY_LENGTH
                        ):
                            user_history.pop(0)

        for j in range(len(user_history)):
            encoded_history += (
                ">".join(
                    [user_history[j].utterance, "_".join(user_history[j].dialogue_acts)]
                )
                + "|"
            )

        for j in range(len(agent_history)):
            encoded_history += (
                ">".join(
                    [
                        agent_history[j].utterance,
                        "_".join(agent_history[j].dialogue_acts),
                        "_".join(agent_history[j].to_be_retrieved),
                    ]
                )
                + "|"
            )

        # Get the last user utterance and dialogue acts
        last_user_utterance = dialog_item.utterance
        last_user_dialogue_act = dialog_item.dialogue_acts
        encoded_history = (
            encoded_history
            + "_".join(last_user_dialogue_act)
            + ">"
            + last_user_utterance
        )

        return encoded_history

    def predict(self, encoded_history):
        """
        Given an encoded history, predicts the dialogue acts of the last turn.
        """
        self.model.eval()

        # Separate the history from the current utterance splitting by the last "|" character, but don't remove it
        history, utterance = encoded_history.rsplit(">", 1)

        # Merge the history and sentence into a single string adding a [SEP] token between them
        encoded_history = "".join(history) + " [SEP] " + utterance

        encoded = self.tokenizer.encode_plus(
            encoded_history,
            add_special_tokens=True,
            max_length=256,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
        )

        input_ids = torch.tensor([encoded["input_ids"]], dtype=torch.long).to(
            self.device
        )
        attention_mask = torch.tensor([encoded["attention_mask"]], dtype=torch.long).to(
            self.device
        )

        # Make a prediction
        with torch.no_grad():
            logits = self.model(input_ids, attention_mask)

        logits_cpu = logits.to("cpu")
        return logits_cpu.numpy()

    def predict_only_last_dialog_item(
        self, dialog_item_dataset: typing.List[DialogItem]
    ) -> typing.List[str]:
        """
        Predicts the dialogue act of last User turn in the dialog. For the Agent, the DAs are filled using the ground truth from the pre-processing function and it arrives here filled, since
        we know what dialog acts the agent is performing. For previous User DAs, the ground truth is also used (allowed by professors)

        Parameters:
        - dialog_item_dataset: List of DialogItem objects.


        Returns:
        - List of predicted dialogue acts
        """

        dataset_copy = dialog_item_dataset.copy()

        # If the speaker is the User, predict the dialogue act
        encoded_history = self.convert_dialogitem_encoded_history(
            dataset_copy[-1], dataset_copy
        )
        preds = self.predict(encoded_history)
        threshold = 0.5
        all_preds_binary = []
        for all_pred in preds:
            local_pred = []
            for old_local_pred in all_pred:
                binary_local_pred = (old_local_pred > threshold).astype(int)
                local_pred.append(binary_local_pred)
            all_preds_binary.append(local_pred)
        labels_preds = self.mlb.inverse_transform(np.array(all_preds_binary))
        dialogue_acts = labels_preds[0]

        return dialogue_acts


# 3.2:


class DAAgentMoveClassifier(nn.Module):
    bert_model_type = "distilbert-base-uncased"

    def __init__(self, num_classes: int):
        super(DAAgentMoveClassifier, self).__init__()
        self.bert = DistilBertModel.from_pretrained(
            DAAgentMoveClassifier.bert_model_type
        )
        self.fc = nn.Linear(768, num_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[
            :, 0, :
        ]  # selects the [CLS] token position.
        logits = torch.sigmoid(self.fc(cls_output))
        return logits


class AgentDAModel:
    PAST_HISTORY_LENGTH = 1
    bert_model_type = "distilbert-base-uncased"

    def __init__(self, model_path="", mlb_path=""):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = DistilBertTokenizer.from_pretrained(
            AgentDAModel.bert_model_type
        )
        self.mlb: MultiLabelBinarizer = pickle.load(open(mlb_path, "rb"))
        # Load the best model
        saved_model = DAAgentMoveClassifier(num_classes=len(self.mlb.classes_)).to(
            self.device
        )
        best_model = torch.load(model_path, map_location=torch.device(self.device))
        saved_model.load_state_dict(best_model["model_state_dict"])

        self.model = saved_model

    def convert_dialogitem_encoded_history(
        self,
        dialog_item: DialogItemForAgentMove,
        dialog_item_dataset: typing.List[DialogItemForAgentMove],
        to_be_provided_gt: typing.List[str],
    ):
        """
        Converts a DialogItem object into an Encoded History string.

        Parameters:
        - dialog_item: DialogItem object.
        - dialog_item_dataset: List of DialogItem objects.
        - past_history_length: Length of the past history to consider.


        Returns:
        - Encoded History string.
        """
        encoded_history = ""

        # Initialize the history of the user and agent as lists with empty DialogItem objects
        agent_history: typing.List[DialogItemForAgentMove] = []
        user_history: typing.List[DialogItemForAgentMove] = []

        # For each identifier of the previous dialog items, search in the dialog_item_dataset for the corresponding DialogItem object
        # and append it to the agent or user history. If no DialogItem object is found, append a dialog item with empty utterance "" and dialogue acts []
        for i in range(len(dialog_item.previous_dialog_items)):
            for j in range(len(dialog_item_dataset)):
                if (
                    dialog_item_dataset[j].id_dialog
                    == dialog_item.previous_dialog_items[i].id_dialog
                    and dialog_item_dataset[j].order_in_dialog
                    == dialog_item.previous_dialog_items[i].order_in_dialog
                ):
                    if dialog_item_dataset[j].speaker == "Agent":
                        agent_history.append(dialog_item_dataset[j])
                        # If the agent history is longer than the past history length, remove the oldest turn
                        if (
                            len(agent_history)
                            > AgentToBeRetrievedModel.PAST_HISTORY_LENGTH
                        ):
                            agent_history.pop(0)
                    elif dialog_item_dataset[j].speaker == "User":
                        user_history.append(dialog_item_dataset[j])
                        # If the user history is longer than the past history length, remove the oldest turn
                        if (
                            len(user_history)
                            > AgentToBeRetrievedModel.PAST_HISTORY_LENGTH
                        ):
                            user_history.pop(0)

        # Get the last user utterance and dialogue acts
        last_user_utterance = dialog_item.utterance
        last_user_dialogue_act = dialog_item.dialogue_acts
        encoded_history = (
            encoded_history
            + last_user_utterance
            + ">"
            + "_".join(last_user_dialogue_act)
            + "|"
        )
        for j in range(len(user_history)):
            encoded_history += (
                ">".join(
                    [user_history[j].utterance, "_".join(user_history[j].dialogue_acts)]
                )
                + "|"
            )
        for j in range(len(agent_history)):
            encoded_history += (
                ">".join(
                    [
                        agent_history[j].utterance,
                        "_".join(agent_history[j].dialogue_acts),
                    ]
                )
                + "|"
            )

        # Add the to be provided ground truth to the encoded history for this turn
        encoded_history = encoded_history + "_".join(to_be_provided_gt)

        return encoded_history

    def predict(self, encoded_history):
        """
        Given an encoded history, predicts the dialogue acts of the last turn.
        """
        self.model.eval()

        # Separate the history from the current utterance splitting by the last "|" character, but don't remove it
        history, utterance = encoded_history.rsplit(">", 1)

        # Merge the history and sentence into a single string adding a [SEP] token between them
        encoded_history = "".join(history) + " [SEP] " + utterance

        encoded = self.tokenizer.encode_plus(
            encoded_history,
            add_special_tokens=True,
            max_length=256,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
        )

        input_ids = torch.tensor([encoded["input_ids"]], dtype=torch.long).to(
            self.device
        )
        attention_mask = torch.tensor([encoded["attention_mask"]], dtype=torch.long).to(
            self.device
        )

        # Make a prediction
        with torch.no_grad():
            logits = self.model(input_ids, attention_mask)

        logits_cpu = logits.to("cpu")
        return logits_cpu.numpy()

    def predict_only_last_dialog_item(
        self,
        dialog_item_dataset: typing.List[DialogItem],
        to_be_provided_gt: typing.List[str],
    ) -> typing.List[str]:
        """
        Predicts the dialogue act of last User turn in the dialog. For the Agent, the DAs are filled using the ground truth from the pre-processing function and it arrives here filled, since
        we know what dialog acts the agent is performing. For previous User DAs, the ground truth is also used (allowed by professors)

        Parameters:
        - dialog_item_dataset: List of DialogItem objects.


        Returns:
        - List of predicted dialogue acts
        """

        dataset_copy = dialog_item_dataset.copy()

        # If the speaker is the User, predict the dialogue act
        encoded_history = self.convert_dialogitem_encoded_history(
            dataset_copy[-1], dataset_copy, to_be_provided_gt
        )
        preds = self.predict(encoded_history)
        threshold = 0.5
        all_preds_binary = []
        for all_pred in preds:
            local_pred = []
            for old_local_pred in all_pred:
                binary_local_pred = (old_local_pred > threshold).astype(int)
                local_pred.append(binary_local_pred)
            all_preds_binary.append(local_pred)
        labels_preds = self.mlb.inverse_transform(np.array(all_preds_binary))
        dialogue_acts = labels_preds[0]

        return dialogue_acts


# 3.3:


class ToBeProvided:
    # Load the dataset
    def load_dataset():
        dataset = load_dataset("multi_woz_v22")
        train_data = dataset["train"]
        val_data = dataset["validation"]
        test_data = dataset["test"]
        return train_data, val_data, test_data

    # Get the relevant data
    def filterDomains(data):
        return [
            entry
            for entry in data
            if set(entry["services"]).issubset({"restaurant", "hotel", "booking"})
        ]

    # Add the data to be retrieved
    def add_data_to_be_retrieved(dataset, print_dialogue=False):
        """
        Augment the dataset with the following information:
        - Information to be retrieved (ground truth)

        Heavily inspired by the code from the evaluation script.
        """

        for dialogue in dataset:
            turns = dialogue["turns"]
            turns["to_be_retrieved_ground_truth"] = {
                turn_id: [] for turn_id in range(len(turns["turn_id"]))
            }

            for turn_id, _ in enumerate(turns["turn_id"]):
                # If it is SYSTEM turn:
                if turns["speaker"][turn_id]:
                    slot_names_per_act = [
                        slot["slot_name"]
                        for slot in turns["dialogue_acts"][turn_id]["dialog_act"][
                            "act_slots"
                        ]
                    ]
                    slot_values_per_act = [
                        slot["slot_value"]
                        for slot in turns["dialogue_acts"][turn_id]["dialog_act"][
                            "act_slots"
                        ]
                    ]
                    dialogue_acts = turns["dialogue_acts"][turn_id]["dialog_act"][
                        "act_type"
                    ]
                    services = turns["frames"][turn_id]["service"]
                    current_booking_service = [
                        service
                        for service in services
                        if service in ["hotel", "restaurant"]
                    ]

                    to_be_retrieved_ground_truth = []
                    for act_i in range(len(slot_names_per_act)):
                        domain = dialogue_acts[act_i].split("-")[0].lower()
                        if domain == "booking" and len(current_booking_service) == 1:
                            domain = current_booking_service[0]
                        slot_names = [
                            domain + "-" + slot_names_per_act[act_i][slot_i]
                            for slot_i in range(len(slot_names_per_act[act_i]))
                            if slot_values_per_act[act_i][slot_i] != "?"
                            and slot_names_per_act[act_i][slot_i] != "none"
                        ]
                        if slot_names:
                            to_be_retrieved_slot_names = [
                                "%s-availability" % (domain)
                            ] + slot_names
                            while domain + "-choice" in to_be_retrieved_slot_names:
                                del to_be_retrieved_slot_names[
                                    to_be_retrieved_slot_names.index(domain + "-choice")
                                ]
                            to_be_retrieved_ground_truth.extend(
                                to_be_retrieved_slot_names
                            )
                    to_be_retrieved_ground_truth = sorted(
                        list(set(to_be_retrieved_ground_truth))
                    )

                    # augment the dataset
                    turns["to_be_retrieved_ground_truth"][turn_id].extend(
                        to_be_retrieved_ground_truth
                    )

                    if print_dialogue:
                        print(f"Utterance: {turns['utterance'][turn_id]}")
                        print(f"To be retrieved: {to_be_retrieved_ground_truth}")
            if print_dialogue:
                print("-" * 50)

    # Add the data to be provided
    def add_data_to_be_provided(dataset):
        """
        Augment the dataset with the following information:
        - Information to be provided (ground truth)

        Heavily inspired by the code from the evaluation script.
        """
        for dialogue in dataset:
            turns = dialogue["turns"]
            turns["to_be_provided_overall"] = {
                turn_id: [] for turn_id in range(len(turns["turn_id"]))
            }

            for turn_id, _ in enumerate(turns["turn_id"]):
                # If it is SYSTEM turn:
                if turns["speaker"][turn_id]:
                    slot_names_per_act = [
                        slot["slot_name"]
                        for slot in turns["dialogue_acts"][turn_id]["dialog_act"][
                            "act_slots"
                        ]
                    ]
                    slot_values_per_act = [
                        slot["slot_value"]
                        for slot in turns["dialogue_acts"][turn_id]["dialog_act"][
                            "act_slots"
                        ]
                    ]
                    dialogue_acts = turns["dialogue_acts"][turn_id]["dialog_act"][
                        "act_type"
                    ]
                    services = turns["frames"][turn_id]["service"]
                    current_booking_service = [
                        service
                        for service in services
                        if service in ["hotel", "restaurant"]
                    ]
                    to_be_provided_overall = []

                    for act_i in range(len(slot_names_per_act)):
                        domain = dialogue_acts[act_i].split("-")[0].lower()
                        if domain == "booking" and len(current_booking_service) == 1:
                            domain = current_booking_service[0]
                        if domain in ["hotel", "restaurant", "booking", "general"]:
                            slot_names_vlues = [
                                domain
                                + "-"
                                + slot_names_per_act[act_i][slot_i]
                                + ":"
                                + slot_values_per_act[act_i][slot_i]
                                for slot_i in range(len(slot_names_per_act[act_i]))
                                if slot_values_per_act[act_i][slot_i] != "?"
                                and slot_names_per_act[act_i][slot_i] != "none"
                            ]
                            if (
                                slot_names_vlues
                                and any(
                                    (
                                        slot_name_value.split(":")[0]
                                        != domain + "-none"
                                        for slot_name_value in slot_names_vlues
                                    )
                                )
                                and not "-No" in dialogue_acts[act_i]
                            ):
                                to_be_provided = [
                                    "%s-availability:yes" % (domain)
                                ] + slot_names_vlues
                                to_be_provided_overall.extend(to_be_provided)
                            elif "-No" in dialogue_acts[act_i]:
                                to_be_provided = [
                                    "%s-availability:no" % (domain)
                                ] + slot_names_vlues
                                to_be_provided_overall.extend(to_be_provided)
                    to_be_provided_overall = sorted(list(set(to_be_provided_overall)))
                    remove_avail_no_list = [
                        elem
                        for elem in to_be_provided_overall
                        if elem.endswith("availability:no")
                    ]
                    for remove_avail in remove_avail_no_list:
                        remove_avail_yes = remove_avail[:-2] + "yes"
                        while remove_avail_yes in to_be_provided_overall:
                            del to_be_provided_overall[
                                to_be_provided_overall.index(remove_avail_yes)
                            ]
                    turns["to_be_provided_overall"][turn_id].extend(
                        to_be_provided_overall
                    )

    def extract_relevant_data(dataset):
        """
        Create a dataset that can be used for using the what shall be requested model.
        """

        user_dialogue_acts = []
        extracted_information = []
        retrieved_information = []
        information_to_be_requested = []

        for dialogue in dataset:
            turns = dialogue["turns"]

            for turn_id, _ in enumerate(turns["turn_id"]):
                # if it is the USER turn:
                if not turns["speaker"][turn_id]:
                    user_dialogue_acts.append(
                        turns["dialogue_acts"][turn_id]["dialog_act"]["act_type"]
                    )
                    slot_names_per_act = [
                        slot["slot_name"]
                        for slot in turns["dialogue_acts"][turn_id]["dialog_act"][
                            "act_slots"
                        ]
                    ]
                    slot_values_per_act = [
                        slot["slot_value"]
                        for slot in turns["dialogue_acts"][turn_id]["dialog_act"][
                            "act_slots"
                        ]
                    ]
                    current_slots = []
                    for act_i in range(len(slot_names_per_act)):
                        for slot_name, slot_value in zip(
                            slot_names_per_act[act_i], slot_values_per_act[act_i]
                        ):
                            if slot_name != "none":
                                current_slots.append(slot_name + ":" + slot_value)
                    extracted_information.append(current_slots)

                # If it is SYSTEM turn:
                if turns["speaker"][turn_id]:
                    retrieved_information.append(
                        turns["to_be_provided_overall"][turn_id]
                    )
                    # get the slot names with '?' as value
                    agent_dialogue_acts = turns["dialogue_acts"][turn_id]["dialog_act"][
                        "act_type"
                    ]
                    slot_names_per_act = [
                        slot["slot_name"]
                        for slot in turns["dialogue_acts"][turn_id]["dialog_act"][
                            "act_slots"
                        ]
                    ]
                    slot_values_per_act = [
                        slot["slot_value"]
                        for slot in turns["dialogue_acts"][turn_id]["dialog_act"][
                            "act_slots"
                        ]
                    ]
                    slots_to_be_requested = []
                    for act_i in range(len(slot_names_per_act)):
                        for slot_name, slot_value in zip(
                            slot_names_per_act[act_i], slot_values_per_act[act_i]
                        ):
                            if slot_value == "?":
                                prefix = (
                                    agent_dialogue_acts[act_i - 1].split("-")[0].lower()
                                    + "-"
                                )
                                slots_to_be_requested.append(prefix + slot_name)
                    information_to_be_requested.append(slots_to_be_requested)

        model_dataset = {
            "user_dialogue_acts": user_dialogue_acts,
            "extracted_information": extracted_information,
            "retrieved_information": retrieved_information,
            "information_to_be_requested": information_to_be_requested,
        }
        return model_dataset

    def create_x_y(dataset):
        """
        Create the x and y for the model.
        """
        x = []
        y = []
        for i in range(len(dataset["user_dialogue_acts"])):
            x.append(
                dataset["user_dialogue_acts"][i]
                + dataset["extracted_information"][i]
                + dataset["retrieved_information"][i]
            )
            y.append(dataset["information_to_be_requested"][i]) if dataset[
                "information_to_be_requested"
            ][i] else y.append(["none"])
        return x, y

    def undersample_none_label(
        x_data, y_data, undersample_ratio=1.0, random_state=None
    ):
        """
        Undersamples the 'none' label in a multi-label dataset.

        :param x_data: Feature set, list of lists or similar.
        :param y_data: Label set, list of lists or lists of sets of labels.
        :param undersample_ratio: Ratio of number of 'none' instances to other instances (default: 1.0).
        :param random_state: Integer seed for reproducibility (default: None).
        :return: Tuple of undersampled (x_data, y_data).
        """
        if random_state is not None:
            random.seed(random_state)

        # Convert labels to a set for easier manipulation
        y_data_sets = [set(labels) for labels in y_data]

        # Separate 'none' instances and other instances
        none_indices = [i for i, labels in enumerate(y_data_sets) if labels == {"none"}]
        other_indices = [
            i for i, labels in enumerate(y_data_sets) if labels != {"none"}
        ]

        # Calculate the number of 'none' instances to keep
        num_none_to_keep = int(len(other_indices) * undersample_ratio)

        # Randomly undersample 'none' instances
        random.shuffle(none_indices)
        none_indices = none_indices[:num_none_to_keep]

        # Combine back the indices
        undersampled_indices = none_indices + other_indices

        # Subset the original x_data and y_data
        x_data_undersampled = [x_data[i] for i in undersampled_indices]
        y_data_undersampled = [y_data[i] for i in undersampled_indices]

        return x_data_undersampled, y_data_undersampled

    def prepare_data_for_bert(x_data, y_data, max_length=384):
        """
        Prepare the data for BERT.
        """

        input_ids = []
        attention_masks = []
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

        for x in x_data:
            x_joined = " ".join(x)

            # Optional: Check the length after joining
            if len(x_joined) > max_length:
                print(
                    "Warning: Truncating input with length %d to max_length %d."
                    % (len(x_joined), max_length)
                )

            x_encoded = tokenizer.encode_plus(
                x_joined,
                max_length=max_length,
                add_special_tokens=True,
                return_token_type_ids=False,
                padding="max_length",
                truncation=True,
                return_attention_mask=True,
                return_tensors="pt",
            )
            input_ids.append(x_encoded["input_ids"])
            attention_masks.append(x_encoded["attention_mask"])

        # Convert lists to tensors
        input_ids = torch.cat(input_ids, dim=0)
        attention_masks = torch.cat(attention_masks, dim=0)
        y_data_tensor = torch.tensor(y_data)

        return input_ids, attention_masks, y_data_tensor

    class CustomDataset(Dataset):
        def __init__(self, input_ids, attention_masks, labels):
            self.input_ids = input_ids
            self.attention_masks = attention_masks
            self.labels = labels

        def __len__(self):
            return len(self.labels)

        def __getitem__(self, idx):
            return self.input_ids[idx], self.attention_masks[idx], self.labels[idx]

    class BertForMultiLabelClassification(nn.Module):
        def __init__(self, num_labels):
            super(ToBeProvided.BertForMultiLabelClassification, self).__init__()
            self.num_labels = num_labels
            self.bert = BertModel.from_pretrained("bert-base-uncased")
            self.dropout = nn.Dropout(0.1)
            self.custom_layer = nn.Linear(
                self.bert.config.hidden_size, self.bert.config.hidden_size
            )
            self.custom_activation = nn.ReLU()
            self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)

        def forward(self, input_ids, attention_mask):
            outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
            pooled_output = outputs.pooler_output
            pooled_output = self.dropout(pooled_output)
            custom_output = self.custom_activation(self.custom_layer(pooled_output))
            logits = self.classifier(custom_output)
            return logits

    class EarlyStopping:
        def __init__(self, patience=3, verbose=False, delta=0):
            self.patience = patience
            self.verbose = verbose
            self.counter = 0
            self.best_score = None
            self.early_stop = False
            self.val_loss_min = np.Inf
            self.delta = delta

        def __call__(self, val_loss, model):
            score = -val_loss

            if self.best_score is None:
                self.best_score = score
                self.save_checkpoint(val_loss, model)
            elif score < self.best_score + self.delta:
                self.counter += 1
                if self.verbose:
                    print(
                        f"EarlyStopping counter: {self.counter} out of {self.patience}"
                    )
                if self.counter >= self.patience:
                    self.early_stop = True
            else:
                self.best_score = score
                self.save_checkpoint(val_loss, model)
                self.counter = 0

        def save_checkpoint(self, val_loss, model):
            if self.verbose:
                print(
                    f"Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model ..."
                )
            torch.save(model.state_dict(), "checkpoint.pt")
            self.val_loss_min = val_loss

    def train_model(
        model,
        train_dataloader,
        val_dataloader,
        optimizer,
        criterion,
        num_epochs,
        device,
    ):
        early_stopping = ToBeProvided.EarlyStopping(patience=2, verbose=True)

        for epoch in range(num_epochs):
            model.train()
            total_loss = 0

            for batch in train_dataloader:
                b_input_ids, b_attention_mask, b_labels = [t.to(device) for t in batch]

                optimizer.zero_grad()
                outputs = model(b_input_ids, b_attention_mask)
                loss = criterion(outputs, b_labels.float())
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            avg_train_loss = total_loss / len(train_dataloader)
            print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}")

            # Validation phase
            model.eval()
            total_val_loss = 0
            with torch.no_grad():
                for batch in val_dataloader:
                    b_input_ids, b_attention_mask, b_labels = [
                        t.to(device) for t in batch
                    ]

                    outputs = model(b_input_ids, b_attention_mask)
                    loss = criterion(outputs, b_labels.float())
                    total_val_loss += loss.item()

            avg_val_loss = total_val_loss / len(val_dataloader)
            print(f"Epoch {epoch+1}/{num_epochs}, Val Loss: {avg_val_loss:.4f}")

            # Early Stopping
            early_stopping(avg_val_loss, model)
            if early_stopping.early_stop:
                print("Early stopping")
                break

    def evaluate_model(
        model, test_dataloader, device, threshold=0.5, path_to_model=None
    ):
        if path_to_model:
            model.load_state_dict(torch.load(path_to_model))
        model.eval()

        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch in test_dataloader:
                b_input_ids, b_attention_mask, b_labels = [t.to(device) for t in batch]

                outputs = model(b_input_ids, b_attention_mask)
                preds = torch.sigmoid(outputs).cpu().numpy()
                labels = b_labels.cpu().numpy()

                all_preds.append(preds)
                all_labels.append(labels)

        # Flatten the outputs and labels lists
        all_preds = np.vstack(all_preds)
        all_labels = np.vstack(all_labels)

        # Apply threshold to convert probabilities to binary predictions
        all_preds_binary = (all_preds > threshold).astype(int)

        return all_preds_binary, all_labels

    def process_input_for_query(user_input, tokenizer, max_length=384):
        # Tokenize and encode the input text
        encoded_input = tokenizer.encode_plus(
            user_input,
            max_length=max_length,
            add_special_tokens=True,
            return_token_type_ids=False,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        return encoded_input["input_ids"], encoded_input["attention_mask"]

    def load_trained_model(model_path, mlb_path):
        # load the MultiLabelBinarizer
        mlb = joblib.load(mlb_path)

        # Get the number of labels
        num_labels = len(mlb.classes_)

        # Initialize the model structure
        model = ToBeProvided.BertForMultiLabelClassification(num_labels=num_labels)
        # Load the trained weights
        model.load_state_dict(torch.load(model_path))
        return model

    def query_model(user_input, model, tokenizer, mlb, device, threshold=0.5):
        # Process the input
        input_ids, attention_mask = ToBeProvided.process_input_for_query(
            user_input, tokenizer
        )

        # Load the trained model
        model.to(device)
        model.eval()

        # Make prediction
        with torch.no_grad():
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            logits = model(input_ids, attention_mask)
            probabilities = torch.sigmoid(logits).cpu().numpy()

        # Apply threshold to get binary predictions
        predictions = (probabilities > threshold).astype(int)

        # Transform binary predictions back to label names
        label_names = mlb.inverse_transform(predictions)

        return label_names


# Slot Filling Model
slot_filling_model = return_model("slot_filling_model.pt")

# Dialogue Act Prediction Model
dialogue_act_testing_suite = DialogActModel(
    mlb_path="da_mlb.pkl",
    model_path="dialog_act_model.pth",
    bert_model_type="distilbert-base-uncased",
)

# Agent to be retrieved model
agent_to_be_retrieved_testing_suite = AgentToBeRetrievedModel(
    mlb_path="mlb3.1.pkl", model_path="best_model3.1.pth"
)

# Agent DA model
agent_da_testing_suite = AgentDAModel(
    mlb_path="mlb3.2.pkl", model_path="best_model3.2.pth"
)

# To be provided model
to_be_provided_testing_suite = ToBeProvided.load_trained_model(
    "to_be_provided_model.pt", "mlb3-3.pkl"
)


def Dialogue_Act_Prediction(user_utterance, other_features_from_dialogue_history):
    # Make copies so it doenst change the original, which will be changed after prediction with ground truth values
    dialogue_history_copy = other_features_from_dialogue_history[
        "dialogue_history"
    ].copy()

    previous_dialog_history_ids_copy = other_features_from_dialogue_history[
        "previous_dialogue_history_ids"
    ].copy()

    # Add the user utterance to the dialogue history with the necessary features
    DialogActModel.add_dialogue_items_to_dialogue_history(
        utterance=user_utterance,
        speaker=other_features_from_dialogue_history["speaker"],
        dialog_acts=[],
        id_dialog=other_features_from_dialogue_history["id_dialogue"],
        order_in_dialog=other_features_from_dialogue_history["turn_id"],
        previous_dialog_history_ids=previous_dialog_history_ids_copy,
        dialog_history=dialogue_history_copy,
    )

    # Predict the dialogue act
    dialogue_acts = dialogue_act_testing_suite.predict_only_last_dialog_item(
        dialog_item_dataset=dialogue_history_copy,
    )
    return dialogue_acts


def Extract_and_Categorize_Spans(
    user_utterance, user_dialogue_acts, other_features_from_dialogue_history2={}
):
    parser = SlotValueParser()
    predicted_slots = slot_filling_model.query_slots(user_utterance)
    extracted_information = []
    for slot_name, predicted_value in predicted_slots.items():
        # Convert the predicted slot value
        predicted_value = parser.convert_span_to_slot_value(
            slot_name,
            predicted_value,
            other_features_from_dialogue_history2["slot_history"],
        )
        extracted_information.append((slot_name, predicted_value))
    # extracted_information = [('hotel-bookpeople', '2'), ('hotel-bookstay', '2'), ('hotel-bookday', 'sunday'), ('restaurant-phone', '?')]
    return extracted_information


def Information_to_be_retrieved_Prediction(
    user_dialogue_acts, extracted_information, other_features_from_dialogue_history3
):
    # Predict to be retrieved
    to_be_retrieved = agent_to_be_retrieved_testing_suite.predict_only_last_dialog_item(
        dialog_item_dataset=other_features_from_dialogue_history3["dialogue_history"],
    )
    to_be_retrieved = list(to_be_retrieved)
    return to_be_retrieved


def Agent_Move_Prediction(
    user_dialogue_acts,
    extracted_information,
    retrieved_information,
    other_features_from_dialogue_history4,
):
    # Model 3.2
    # Predict the dialogue act
    # Filter retrieved information to only have the part before :
    retrieved_information_filtered = []
    for i in range(len(retrieved_information)):
        retrieved_information_filtered.append(retrieved_information[i].split(":")[0])
    agent_dialogue_acts = agent_da_testing_suite.predict_only_last_dialog_item(
        dialog_item_dataset=other_features_from_dialogue_history4["dialogue_history"],
        to_be_provided_gt=retrieved_information_filtered,
    )
    agent_dialogue_acts = list(agent_dialogue_acts)

    # convert dict to 'key:value' list

    extracted_information = [
        f"{act_list[0]}:{act_list[1]}"
        for act_list in extracted_information
    ]

    input_text = " ".join(user_dialogue_acts + extracted_information)
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    mlb = joblib.load("mlb3-3.pkl")
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    to_be_requested = ToBeProvided.query_model(
        input_text, to_be_provided_testing_suite, tokenizer, mlb, device, threshold=0.5
    )
    
    to_be_requested = [item for tup in to_be_requested for item in tup]
    return {
        "agent_dialogue_acts": agent_dialogue_acts,
        "to_be_requested": to_be_requested,
    }


def print_output(*output):
    if do_print_dialogue_details:
        if len(output) == 1:
            print(output[0])
        elif len(output) == 2:
            print(output[0], output[1])
        elif len(output) == 3:
            print(output[0], output[1], output[2])
        elif len(output) == 4:
            print(output[0], output[1], output[2], output[3])
        else:
            print(output)


def count_matches(ground_truth_list, predicted_list):
    no_gt = len(ground_truth_list)
    no_predicted = len(predicted_list)
    no_correct = no_gt - sum(
        (Counter(ground_truth_list) - Counter(predicted_list)).values()
    )
    return no_gt, no_predicted, no_correct


def get_metrics(no_gt_global, no_predicted_global, no_correct_global):
    precision = (
        1.0 * no_correct_global / no_predicted_global if no_predicted_global else 0.0
    )
    recall = 1.0 * no_correct_global / no_gt_global if no_gt_global else 0.0
    f1_score = (
        2.0 * precision * recall / (precision + recall) if precision and recall else 0.0
    )
    return precision, recall, f1_score


dataset = load_dataset("multi_woz_v22")


def eval_fn(
    print_dialogue_details=False,
    print_ground_truth_structures=False,
    print_predicted=False,
):
    global do_print_dialogue_details

    if print_dialogue_details:
        do_print_dialogue_details = True
    else:
        do_print_dialogue_details = False

    no_user_moves_gt_global = 0  # "gt" == "ground truth"
    no_user_moves_predicted_global = 0
    no_user_moves_correct_global = 0

    no_user_info_gt_global = 0
    no_user_info_predicted_global = 0
    no_user_info_correct_global = 0

    no_agent_info_to_retrieve_gt_global = 0
    no_agent_info_to_retrieve_predicted_global = 0
    no_agent_info_to_retrieve_correct_global = 0

    no_agent_moves_gt_global = 0
    no_agent_moves_predicted_global = 0
    no_agent_moves_correct_global = 0

    no_agent_info_to_request_gt_global = 0
    no_agent_info_to_request_predicted_global = 0
    no_agent_info_to_request_correct_global = 0

    n_dialogues_to_evaluate = 10
    n_evaluated = 0
    for d, dial in enumerate(dataset["train"]):
        if n_evaluated >= n_dialogues_to_evaluate:
            break
        # skip dialogues that are not in the hotel or restaurant domain
        if not any(
            set(dial["turns"]["frames"][turn_id]["service"]).intersection(
                ["hotel", "restaurant"]
            )
            for turn_id, utt in enumerate(dial["turns"]["utterance"])
        ):
            continue
        print_output("Dialogue ID:", dial["dialogue_id"])

        # Keep track of dialogue slot history:
        dialogue_history = DialogSlotMemory()

        # Keep track of Dialog Items for Dialogue Act Prediction:
        dialogue_history_for_da_prediction: typing.List[DialogItem] = []
        previous_da_dialog_history_ids: typing.List[DialogItemIdentifier] = []

        # Keep Track of Dialog Items for Agent Slots to Be Retrieved Prediction:
        dialogue_history_for_agent_move_prediction: typing.List[
            DialogItemForAgentMove
        ] = []
        previous_agent_move_dialog_history_ids: typing.List[DialogItemIdentifier] = []

        compulsory_slots_hotel = set(
            ["hotel-bookpeople", "hotel-bookstay", "hotel-name", "hotel-bookday"]
        )  # as an example, to be adjusted
        compulsory_slots_restaurant = set(
            ["restaurant-name"]
        )  # as an example, to be adjusted
        filled_slots = set()
        speaker_str = {0: "User", 1: "Agent"}
        turns = dial["turns"]
        for turn_id, utt in enumerate(turns["utterance"]):
            speaker = speaker_str[turns["speaker"][turn_id]]

            if speaker == "User":
                print_output("User's turn")
                print_output("User's utterance: " + utt)

                print_output("Extraction")

                indent = " " * 4
                dialogue_acts = turns["dialogue_acts"][turn_id]["dialog_act"][
                    "act_type"
                ]
                user_dialogue_acts_ground_truth = dialogue_acts.copy()
                print_output(indent, "Dialogue acts:", dialogue_acts)

                if print_ground_truth_structures:
                    print(user_dialogue_acts_ground_truth)

                if user_dialogue_acts_ground_truth:
                    dialogue_acts_predicted = Dialogue_Act_Prediction(
                        utt,
                        other_features_from_dialogue_history={
                            "dialogue_history": dialogue_history_for_da_prediction,
                            "speaker": speaker,
                            "id_dialogue": dial["dialogue_id"],
                            "turn_id": turn_id,
                            "previous_dialogue_history_ids": previous_da_dialog_history_ids,
                        },
                    )
                    # evaluate user's dialogue acts
                    no_gt, no_predicted, no_correct = count_matches(
                        user_dialogue_acts_ground_truth, dialogue_acts_predicted
                    )
                    no_user_moves_gt_global += no_gt
                    no_user_moves_predicted_global += no_predicted
                    no_user_moves_correct_global += no_correct
                    if print_predicted:
                        print(
                            indent,
                            "User's dialogue acts predicted:",
                            dialogue_acts_predicted,
                            "correct = %d/%d, true covered = %d/%d"
                            % (no_correct, no_predicted, no_correct, no_gt),
                        )

                    # Add the current dialogue act to the dialogue history, AFTER PREDICTION, with the ground truth labels
                    DialogActModel.add_dialogue_items_to_dialogue_history(
                        utterance=utt,
                        speaker=speaker,
                        dialog_history=dialogue_history_for_da_prediction,
                        dialog_acts=dialogue_acts,
                        id_dialog=dial["dialogue_id"],
                        order_in_dialog=turn_id,
                        previous_dialog_history_ids=previous_da_dialog_history_ids,
                    )

                    add_dialogue_items_to_dialogue_history_for_agent_move(
                        utterance=utt,
                        speaker=speaker,
                        dialog_acts=dialogue_acts,
                        to_be_retrieved_gt=[],
                        id_dialog=dial["dialogue_id"],
                        order_in_dialog=turn_id,
                        previous_dialog_history_ids=previous_agent_move_dialog_history_ids,
                        dialog_history=dialogue_history_for_agent_move_prediction,
                    )

                print_output(indent, "Extracted information:")
                print_output(indent, "Spans")
                extracted_information_not_mapped_ground_truth = []
                extracted_information_ground_truth = []
                extracted_information_per_dialogue_act_ground_truth = {}
                span_info = turns["dialogue_acts"][turn_id]["span_info"]
                for span_i in range(len(span_info["span_start"])):
                    act_type = span_info["act_type"][span_i]
                    span_name = span_info["act_slot_name"][span_i]
                    span_value = span_info["act_slot_value"][span_i]
                    span_range = [
                        span_info["span_start"][span_i],
                        span_info["span_end"][span_i],
                    ]
                    span_value_as_in_utterance = utt[
                        span_info["span_start"][span_i] : span_info["span_end"][span_i]
                    ]
                    print_output(
                        indent * 2,
                        span_value
                        + (
                            ""
                            if span_value_as_in_utterance == span_value
                            else " (" + span_value_as_in_utterance + ")"
                        ),
                        span_range,
                    )
                    if (
                        not act_type
                        in extracted_information_per_dialogue_act_ground_truth
                    ):
                        extracted_information_per_dialogue_act_ground_truth[
                            act_type
                        ] = []
                    act_category = act_type.split("-")[0].lower()
                    extracted_information_not_mapped_ground_truth.append(
                        tuple(
                            [act_category + "-" + span_name, span_value_as_in_utterance]
                        )
                    )
                    if act_category in ["hotel", "restaurant", "general"]:
                        extracted_information_ground_truth.append(
                            tuple([act_category + "-" + span_name, span_value])
                        )
                        # update the dialogue history
                        dialogue_history.add_slot(span_name, span_value)
                    extracted_information_per_dialogue_act_ground_truth[
                        act_type
                    ].append(tuple([span_name, span_value]))

                print_output(indent, "Categorized information")
                slot_names_per_act = [
                    slot["slot_name"]
                    for slot in turns["dialogue_acts"][turn_id]["dialog_act"][
                        "act_slots"
                    ]
                ]
                slot_values_per_act = [
                    slot["slot_value"]
                    for slot in turns["dialogue_acts"][turn_id]["dialog_act"][
                        "act_slots"
                    ]
                ]
                for act_i in range(len(slot_names_per_act)):
                    slot_names_values_per_act = [
                        slot_names_per_act[act_i][slot_i]
                        + ":"
                        + slot_values_per_act[act_i][slot_i]
                        for slot_i in range(len(slot_names_per_act[act_i]))
                    ]
                    print_output(
                        indent * 2, dialogue_acts[act_i], slot_names_values_per_act
                    )
                    if dialogue_acts[act_i].startswith("Hotel") or dialogue_acts[
                        act_i
                    ].startswith("Restaurant"):
                        for slot_i in range(len(slot_names_per_act[act_i])):
                            if (
                                slot_names_per_act[act_i][slot_i] != "none"
                                and slot_values_per_act[act_i][slot_i] == "?"
                            ):
                                if (
                                    not dialogue_acts[act_i]
                                    in extracted_information_per_dialogue_act_ground_truth
                                ):
                                    extracted_information_per_dialogue_act_ground_truth[
                                        dialogue_acts[act_i]
                                    ] = []
                                extracted_information_not_mapped_ground_truth.append(
                                    tuple(
                                        [
                                            dialogue_acts[act_i].split("-")[0].lower()
                                            + "-"
                                            + slot_names_per_act[act_i][slot_i],
                                            slot_values_per_act[act_i][slot_i],
                                        ]
                                    )
                                )
                                extracted_information_ground_truth.append(
                                    tuple(
                                        [
                                            dialogue_acts[act_i].split("-")[0].lower()
                                            + "-"
                                            + slot_names_per_act[act_i][slot_i],
                                            slot_values_per_act[act_i][slot_i],
                                        ]
                                    )
                                )
                                extracted_information_per_dialogue_act_ground_truth[
                                    dialogue_acts[act_i]
                                ].append(
                                    tuple(
                                        [
                                            slot_names_per_act[act_i][slot_i],
                                            slot_values_per_act[act_i][slot_i],
                                        ]
                                    )
                                )

                if print_ground_truth_structures:
                    print(extracted_information_not_mapped_ground_truth)
                    print(extracted_information_ground_truth)
                    print(extracted_information_per_dialogue_act_ground_truth)

                if user_dialogue_acts_ground_truth:
                    extracted_information = Extract_and_Categorize_Spans(
                        utt,
                        user_dialogue_acts_ground_truth,
                        other_features_from_dialogue_history2={
                            "slot_history": dialogue_history
                        },
                    )
                    # evaluate information extraction only if there are Hotel or Restaurant or general dialogue acts
                    no_gt, no_predicted, no_correct = count_matches(
                        extracted_information_ground_truth, extracted_information
                    )
                    # print("######################################################")
                    # print(f"Extracted_information_ground_truth: {extracted_information_ground_truth}")
                    ##print(f"Extracted_information: {extracted_information}")
                    # print(no_correct, no_predicted, no_gt)
                    # print("######################################################")
                    print("")
                    if any(
                        da.startswith("general")
                        or da.startswith("Hotel")
                        or da.startswith("Restaurant")
                        for da in user_dialogue_acts_ground_truth
                    ):
                        no_user_info_gt_global += no_gt
                        no_user_info_predicted_global += no_predicted
                        no_user_info_correct_global += no_correct
                        if print_predicted:
                            print(
                                indent,
                                "Extracted and categorized information (predicted):",
                                extracted_information,
                                "correct = %d/%d, true covered = %d/%d"
                                % (no_correct, no_predicted, no_correct, no_gt),
                            )

                print_output("Reasoning (dialogue state tracking)")
                services = turns["frames"][turn_id]["service"]
                print_output(indent, "Services:", services)
                current_booking_service = [
                    service
                    for service in services
                    if service in ["hotel", "restaurant"]
                ]

                not_empty_intents = [
                    intent
                    for intent in turns["frames"][turn_id]["state"]
                    if intent["requested_slots"]
                    or intent["slots_values"]["slots_values_name"]
                ]
                if not_empty_intents:
                    print_output(indent, "Intents")
                    for intent in not_empty_intents:
                        print_output(
                            indent * 2, "Active intent:", intent["active_intent"]
                        )
                        requested_slots = intent["requested_slots"]
                        if requested_slots:
                            print_output(
                                indent * 2, "Requested slots:", requested_slots
                            )
                        if intent["slots_values"]["slots_values_name"]:
                            slot_names = intent["slots_values"]["slots_values_name"]
                            slot_values = intent["slots_values"]["slots_values_list"]
                            filled_slots.update(slot_names)
                            print_output(indent * 2, "Filled slots:")
                            for slot_i in range(len(slot_names)):
                                print_output(
                                    indent * 3,
                                    slot_names[slot_i] + ": ",
                                    slot_values[slot_i],
                                )
                        print_output(indent * 2, "--------------")

                print_output(
                    indent,
                    "Missing slots (Hotel):",
                    compulsory_slots_hotel - filled_slots,
                )
                print_output(
                    indent,
                    "Missing slots (Restaurant):",
                    compulsory_slots_restaurant - filled_slots,
                )
            elif speaker == "Agent":
                indent = " " * 4
                print_output("Agent's turn")
                dialogue_acts = turns["dialogue_acts"][turn_id]["dialog_act"][
                    "act_type"
                ]

                DialogActModel.add_dialogue_items_to_dialogue_history(
                    utterance=utt,
                    speaker=speaker,
                    dialog_acts=dialogue_acts,
                    id_dialog=dial["dialogue_id"],
                    order_in_dialog=turn_id,
                    previous_dialog_history_ids=previous_da_dialog_history_ids,
                    dialog_history=dialogue_history_for_da_prediction,
                )

                do_evaluate_agent_turn = True
                if not any(
                    da.startswith("Hotel")
                    or da.startswith("Restaurant")
                    or da.startswith("Booking")
                    for da in dialogue_acts
                ):
                    do_evaluate_agent_turn = False
                    print_output(
                        "This agent's turn won't be evaluated as it is out of domain."
                    )

                print_output("Retrieval")

                slot_names_per_act = [
                    slot["slot_name"]
                    for slot in turns["dialogue_acts"][turn_id]["dialog_act"][
                        "act_slots"
                    ]
                ]
                slot_values_per_act = [
                    slot["slot_value"]
                    for slot in turns["dialogue_acts"][turn_id]["dialog_act"][
                        "act_slots"
                    ]
                ]

                to_be_retrieved_ground_truth = []
                print_output(indent, "Information to be retrieved:")
                for act_i in range(len(slot_names_per_act)):
                    domain = dialogue_acts[act_i].split("-")[0].lower()
                    if domain == "booking" and len(current_booking_service) == 1:
                        domain = current_booking_service[0]
                    slot_names = [
                        domain + "-" + slot_names_per_act[act_i][slot_i]
                        for slot_i in range(len(slot_names_per_act[act_i]))
                        if slot_values_per_act[act_i][slot_i] != "?"
                        and slot_names_per_act[act_i][slot_i] != "none"
                    ]
                    if slot_names:
                        to_be_retrieved_slot_names = [
                            "%s-availability" % (domain)
                        ] + slot_names
                        while domain + "-choice" in to_be_retrieved_slot_names:
                            del to_be_retrieved_slot_names[
                                to_be_retrieved_slot_names.index(domain + "-choice")
                            ]
                        to_be_retrieved_ground_truth.extend(to_be_retrieved_slot_names)
                to_be_retrieved_ground_truth = sorted(
                    list(set(to_be_retrieved_ground_truth))
                )
                print_output(
                    indent * 2,
                    "To be retrieved:",
                    to_be_retrieved_ground_truth,
                    "<--- That's the first thing we predict in agent's move.",
                )

                to_be_provided_overall = []
                print_output(indent, "Retrieved information:")
                for act_i in range(len(slot_names_per_act)):
                    domain = dialogue_acts[act_i].split("-")[0].lower()
                    if domain == "booking" and len(current_booking_service) == 1:
                        domain = current_booking_service[0]
                    if domain in ["hotel", "restaurant", "booking", "general"]:
                        slot_names_vlues = [
                            domain
                            + "-"
                            + slot_names_per_act[act_i][slot_i]
                            + ":"
                            + slot_values_per_act[act_i][slot_i]
                            for slot_i in range(len(slot_names_per_act[act_i]))
                            if slot_values_per_act[act_i][slot_i] != "?"
                            and slot_names_per_act[act_i][slot_i] != "none"
                        ]
                        if (
                            slot_names_vlues
                            and any(
                                (
                                    slot_name_value.split(":")[0] != domain + "-none"
                                    for slot_name_value in slot_names_vlues
                                )
                            )
                            and not "-No" in dialogue_acts[act_i]
                        ):
                            to_be_provided = [
                                "%s-availability:yes" % (domain)
                            ] + slot_names_vlues
                            to_be_provided_overall.extend(to_be_provided)
                        elif "-No" in dialogue_acts[act_i]:
                            to_be_provided = [
                                "%s-availability:no" % (domain)
                            ] + slot_names_vlues
                            to_be_provided_overall.extend(to_be_provided)
                to_be_provided_overall = sorted(list(set(to_be_provided_overall)))
                remove_avail_no_list = [
                    elem
                    for elem in to_be_provided_overall
                    if elem.endswith("availability:no")
                ]
                for remove_avail in remove_avail_no_list:
                    remove_avail_yes = remove_avail[:-2] + "yes"
                    while remove_avail_yes in to_be_provided_overall:
                        del to_be_provided_overall[
                            to_be_provided_overall.index(remove_avail_yes)
                        ]
                print_output(
                    indent * 2, "Retrieved info to be provided:", to_be_provided_overall
                )

                print_output("Planning")
                agent_dialogue_acts_ground_truth = []
                for act_i in range(len(slot_names_per_act)):
                    domain = dialogue_acts[act_i].split("-")[0].lower()
                    if domain in ["hotel", "restaurant", "booking", "general"]:
                        agent_dialogue_acts_ground_truth.append(dialogue_acts[act_i])
                print_output(
                    indent,
                    "Agent's move (dialogue acts):",
                    agent_dialogue_acts_ground_truth,
                    "<--- That's the second thing we predict in agent's move.",
                )

                print_output(indent, "Information to be requested:")
                to_be_requested_ground_truth = []
                for act_i in range(len(slot_names_per_act)):
                    domain = dialogue_acts[act_i].split("-")[0].lower()
                    if domain == "booking" and len(current_booking_service) == 1:
                        domain = current_booking_service[0]
                    if domain in ["hotel", "restaurant", "booking", "general"]:
                        to_be_requested = [
                            domain + "-" + slot_names_per_act[act_i][slot_i]
                            for slot_i in range(len(slot_names_per_act[act_i]))
                            if slot_values_per_act[act_i][slot_i] == "?"
                        ]
                        to_be_requested_ground_truth.extend(to_be_requested)
                to_be_requested_ground_truth = sorted(
                    list(set(to_be_requested_ground_truth))
                )
                print_output(
                    indent * 2,
                    "To be requested:",
                    to_be_requested_ground_truth,
                    "<--- That's the third thing we predict in agent's move.",
                )

                print_output(
                    indent, "Planned move per dialogue act (we won't evaluate this):"
                )
                for act_i in range(len(slot_names_per_act)):
                    print_output(indent * 2, dialogue_acts[act_i])
                    print_output(
                        indent * 2,
                        "To be provided:",
                        [
                            slot_names_per_act[act_i][slot_i]
                            + ":"
                            + slot_values_per_act[act_i][slot_i]
                            for slot_i in range(len(slot_names_per_act[act_i]))
                            if slot_values_per_act[act_i][slot_i] != "?"
                        ],
                    )
                    print_output(
                        indent * 2,
                        "To be requested:",
                        [
                            slot_names_per_act[act_i][slot_i]
                            + ":"
                            + slot_values_per_act[act_i][slot_i]
                            for slot_i in range(len(slot_names_per_act[act_i]))
                            if slot_values_per_act[act_i][slot_i] == "?"
                        ],
                    )
                    print_output(indent * 2, "--------------")

                agent_move_ground_truth = {
                    "to_be_retrieved": to_be_retrieved_ground_truth,  # set, only unique names of the slots
                    "dialogue_acts": agent_dialogue_acts_ground_truth,
                    "to_be_requested": to_be_requested_ground_truth,  # set, only unique names of the slots
                    "retrieved_information_per_dialogue_act": {},  # non-none names of the slots and values grouped per dialogue act
                }

                for act_i in range(len(slot_names_per_act)):
                    da = dialogue_acts[act_i]
                    if (
                        da.startswith("Hotel")
                        or da.startswith("Restaurant")
                        or da.startswith("Booking")
                        or da.startswith("general")
                    ):
                        for slot_i in range(len(slot_names_per_act[act_i])):
                            if slot_names_per_act[act_i][slot_i] != "none":
                                if (
                                    not dialogue_acts[act_i]
                                    in agent_move_ground_truth[
                                        "retrieved_information_per_dialogue_act"
                                    ]
                                ):
                                    agent_move_ground_truth[
                                        "retrieved_information_per_dialogue_act"
                                    ][dialogue_acts[act_i]] = []
                                agent_move_ground_truth[
                                    "retrieved_information_per_dialogue_act"
                                ][dialogue_acts[act_i]].append(
                                    tuple(
                                        [
                                            slot_names_per_act[act_i][slot_i],
                                            slot_values_per_act[act_i][slot_i],
                                        ]
                                    )
                                )

                if do_evaluate_agent_turn and user_dialogue_acts_ground_truth:
                    agent_to_be_retrieved_predicted = Information_to_be_retrieved_Prediction(
                        user_dialogue_acts_ground_truth,
                        extracted_information_per_dialogue_act_ground_truth,
                        other_features_from_dialogue_history3={
                            "dialogue_history": dialogue_history_for_agent_move_prediction,
                            "speaker": speaker,
                            "id_dialogue": dial["dialogue_id"],
                            "turn_id": turn_id,
                            "previous_dialogue_history_ids": previous_agent_move_dialog_history_ids,
                        },
                    )

                    agent_move_predicted = Agent_Move_Prediction(
                        user_dialogue_acts_ground_truth,
                        extracted_information_per_dialogue_act_ground_truth,
                        to_be_provided_overall,
                        other_features_from_dialogue_history4={
                            "dialogue_history": dialogue_history_for_agent_move_prediction,
                            "speaker": speaker,
                            "id_dialogue": dial["dialogue_id"],
                            "turn_id": turn_id,
                            "previous_dialogue_history_ids": previous_agent_move_dialog_history_ids,
                        },
                    )
                    if print_ground_truth_structures:
                        print(agent_move_ground_truth)

                    if print_predicted:
                        print("Planning predicted")

                    # evaluate "to_be_retrieved" only if there are Hotel or Restaurant or Booking or general dialogue acts
                    no_gt, no_predicted, no_correct = count_matches(
                        agent_move_ground_truth["to_be_retrieved"],
                        agent_to_be_retrieved_predicted,
                    )
                    no_agent_info_to_retrieve_gt_global += no_gt
                    no_agent_info_to_retrieve_predicted_global += no_predicted
                    no_agent_info_to_retrieve_correct_global += no_correct
                    if print_predicted:
                        print(
                            indent,
                            "Info to be retrieved predicted:",
                            agent_to_be_retrieved_predicted,
                            "correct = %d/%d, true covered = %d/%d"
                            % (no_correct, no_predicted, no_correct, no_gt),
                        )

                    # evaluate agent's dialogue acts
                    no_gt, no_predicted, no_correct = count_matches(
                        agent_move_ground_truth["dialogue_acts"],
                        agent_move_predicted["agent_dialogue_acts"],
                    )
                    no_agent_moves_gt_global += no_gt
                    no_agent_moves_predicted_global += no_predicted
                    no_agent_moves_correct_global += no_correct
                    if print_predicted:
                        print(
                            indent,
                            "Agent's dialogue acts predicted:",
                            agent_move_predicted["agent_dialogue_acts"],
                            "correct = %d/%d, true covered = %d/%d"
                            % (no_correct, no_predicted, no_correct, no_gt),
                        )

                    # evaluate "retrieved_information" -> "to_be_requested" only if there are Hotel or Restaurant or Booking or general dialogue acts
                    no_gt, no_predicted, no_correct = count_matches(
                        agent_move_ground_truth["to_be_requested"],
                        agent_move_predicted["to_be_requested"],
                    )
                    tbr_gt = agent_move_ground_truth["to_be_requested"]
                    tbr = agent_move_predicted["to_be_requested"]
                    # print("######################################################")
                    # print(f"To be Requested ground_truth: {tbr_gt}")
                    # print(f"To be Requested predicted: {tbr}")
                    # print("######################################################")
                    no_agent_info_to_request_gt_global += no_gt
                    no_agent_info_to_request_predicted_global += no_predicted
                    no_agent_info_to_request_correct_global += no_correct
                    if print_predicted:
                        print(
                            indent,
                            "Info to be requested predicted:",
                            agent_move_predicted["to_be_requested"],
                            "correct = %d/%d, true covered = %d/%d"
                            % (no_correct, no_predicted, no_correct, no_gt),
                        )

                add_dialogue_items_to_dialogue_history_for_agent_move(
                    utterance=utt,
                    speaker=speaker,
                    dialog_acts=dialogue_acts,
                    to_be_retrieved_gt=agent_move_ground_truth["to_be_retrieved"],
                    id_dialog=dial["dialogue_id"],
                    order_in_dialog=turn_id,
                    previous_dialog_history_ids=previous_agent_move_dialog_history_ids,
                    dialog_history=dialogue_history_for_agent_move_prediction,
                )
                print_output("Agent's utterance: " + utt)
            print_output(
                "-------------------------------------------------------------------"
            )
            print_output(
                "-------------------------------------------------------------------"
            )

        n_evaluated += 1

    print("Dialogue acts in the user's move prediction")
    precision, recall, f1_score = get_metrics(
        no_user_moves_gt_global,
        no_user_moves_predicted_global,
        no_user_moves_correct_global,
    )
    print("Precision: %lf, Recall: %lf, F1-score: %lf" % (precision, recall, f1_score))
    print("Extracted information from user's utterance")
    precision, recall, f1_score = get_metrics(
        no_user_info_gt_global,
        no_user_info_predicted_global,
        no_user_info_correct_global,
    )
    print("Precision: %lf, Recall: %lf, F1-score: %lf" % (precision, recall, f1_score))
    print("Info to be retrieved by the agent")
    precision, recall, f1_score = get_metrics(
        no_agent_info_to_retrieve_gt_global,
        no_agent_info_to_retrieve_predicted_global,
        no_agent_info_to_retrieve_correct_global,
    )
    print("Precision: %lf, Recall: %lf, F1-score: %lf" % (precision, recall, f1_score))
    print("Dialogue acts in the agent's move prediction")
    precision, recall, f1_score = get_metrics(
        no_agent_moves_gt_global,
        no_agent_moves_predicted_global,
        no_agent_moves_correct_global,
    )
    print("Precision: %lf, Recall: %lf, F1-score: %lf" % (precision, recall, f1_score))
    print("Info to be requested by the agent")
    precision, recall, f1_score = get_metrics(
        no_agent_info_to_request_gt_global,
        no_agent_info_to_request_predicted_global,
        no_agent_info_to_request_correct_global,
    )
    print("Precision: %lf, Recall: %lf, F1-score: %lf" % (precision, recall, f1_score))


def manual_query_function():
    # Initialize the necessary components and models
    dialogue_history_for_slot_filling = DialogSlotMemory()
    # Keep track of Dialog Items for Dialogue Act Prediction:
    dialogue_history_for_da_prediction: typing.List[DialogItem] = []
    previous_da_dialog_history_ids: typing.List[DialogItemIdentifier] = []

    # Keep Track of Dialog Items for Agent Slots to Be Retrieved Prediction:
    dialogue_history_for_agent_move_prediction: typing.List[DialogItemForAgentMove] = []
    previous_agent_move_dialog_history_ids: typing.List[DialogItemIdentifier] = []

    while True:
        # Input from the user
        user_input = input("Enter your dialogue or 'exit' to quit: ")
        if user_input.lower() == "exit":
            break

        # --- USER

        # Dialogue Act Prediction
        dialogue_acts_predicted = Dialogue_Act_Prediction(
            user_input, {"dialogue_history": dialogue_history_for_da_prediction, "speaker": "User", "id_dialogue": "manual_query", "turn_id": len(dialogue_history_for_da_prediction), "previous_dialogue_history_ids": previous_da_dialog_history_ids}
        )
        dialogue_acts_predicted = list(dialogue_acts_predicted)

        # Update memory
        DialogActModel.add_dialogue_items_to_dialogue_history(
            utterance=user_input, speaker="User", dialog_history=dialogue_history_for_da_prediction, dialog_acts=dialogue_acts_predicted, id_dialog="manual_query", order_in_dialog=len(dialogue_history_for_da_prediction), previous_dialog_history_ids=previous_da_dialog_history_ids
        )

        add_dialogue_items_to_dialogue_history_for_agent_move(
            utterance=user_input, speaker="User", dialog_acts=dialogue_acts_predicted, to_be_retrieved_gt=[], id_dialog="manual_query", order_in_dialog=len(dialogue_history_for_agent_move_prediction), previous_dialog_history_ids=previous_agent_move_dialog_history_ids, dialog_history=dialogue_history_for_agent_move_prediction
        )

        # Slot Filling

        extracted_information = Extract_and_Categorize_Spans(
            user_input,
            dialogue_acts_predicted,
            {"slot_history": dialogue_history_for_slot_filling},
        )

        # update the dialogue history for slot filling
        for frame in extracted_information:
            dialogue_history_for_slot_filling.add_slot(frame[0], frame[1])
        
        # Agent: Simulate the agent's response based on the prediction

        # Simulate the agent's response based on the prediction
        agent_to_be_retrieved_predicted = Information_to_be_retrieved_Prediction(
            dialogue_acts_predicted,
            extracted_information,
            {"dialogue_history": dialogue_history_for_agent_move_prediction},
        )

        agent_move_predicted = Agent_Move_Prediction(
            dialogue_acts_predicted,
            extracted_information,
            agent_to_be_retrieved_predicted,
            {"dialogue_history": dialogue_history_for_agent_move_prediction},
        )

        # Update the dialogue history (after prediction)
        DialogActModel.add_dialogue_items_to_dialogue_history(
            utterance="",
            speaker="Agent",
            dialog_history=dialogue_history_for_da_prediction,
            dialog_acts=agent_move_predicted["agent_dialogue_acts"],
            id_dialog="manual_query",
            order_in_dialog=len(dialogue_history_for_da_prediction),
            previous_dialog_history_ids=previous_da_dialog_history_ids,
        )
        add_dialogue_items_to_dialogue_history_for_agent_move(
            utterance="",
            speaker="Agent",
            dialog_acts=agent_move_predicted["agent_dialogue_acts"],
            to_be_retrieved_gt=agent_to_be_retrieved_predicted,
            id_dialog="manual_query",
            order_in_dialog=len(dialogue_history_for_da_prediction),
            previous_dialog_history_ids=previous_agent_move_dialog_history_ids,
            dialog_history=dialogue_history_for_agent_move_prediction,
        )

        # Display the results
        print("Predicted Dialogue Acts:", dialogue_acts_predicted)
        print("Extracted Information:", extracted_information)
        print("Agent to Retrieve:", agent_to_be_retrieved_predicted)
        print("Agent's Predicted Move:", agent_move_predicted)

        # Add a separator for readability
        print("------------------------------------------------")


if __name__ == "__main__":
    # eval_fn(print_dialogue_details = False, print_ground_truth_structures = False, print_predicted = False) # Run evaluation notebook code.
    manual_query_function()  # Run manual query function.
