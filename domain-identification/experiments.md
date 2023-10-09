## Experiments

07/10/2023 14:30 (Camile) - Distilled Bert uncased with a max sequence length of 128 with 16 batches and 3 epochs. That resulted in the highest accuracy so far (85.1%) - ON VALIDATION DATA

07/10/2023 17:00 (Bernardo) - Added test data to the training process, training on 10 epochs and a train vs validation loss plot. Getting similar accuracy results on the test data than the previous iteration. Added a confusion matrix to the results and a classification report, and also capability to save the best model (i.e smallest validation loss) when it overfits. 

07/10/2023 18:38 (Bernardo) - Accuracy on Distilled Bert Uncased with a max sequence length of 128 with 64 batches and 4 epochs is 82.2% on the test data. Overfitting instantly.

07/10/2023 19:30 (Bernardo) - Accuracy on Distilled Bert Uncased with a max sequence length of 128 with 64 batches and 10 epochs is 83.1% on the test data after adjusting learning rate to 5e-6. Overfits around epoch 4.

07/10/2023 19:59 (Bernardo) - Accuracy on Distilled Bert Uncased with a max sequence length of 128 with 64 batches and 10 epochs is 83.6% on the test data with learning rate to 5e-6. Overfits around epoch 4.

07/10/2023 20:20 (Bernardo) - Accuracy on Distilled Bert Uncased with a max sequence length of 128 with 32 batches and 10 epochs is 83.5% on the test data with a learning rate to 5e-6. Overfits around epoch 4.

07/10/2023 20:40 (Bernardo) - Accuracy on Distilled Bert Uncased with a max sequence length of 128 with 32 batches and 6 epochs is 81.9% on the test data with a learning rate to 2e-5. Overfits around epoch 2.

07/10/2023 21:00 (Bernardo) - Accuracy on Distilled Bert Uncased with a max sequence length of 128 with 32 batches and 6 epochs is 83.1% on the test data with a learning rate to 2e-6. Overfits around epoch 3.

--> Best model so far is: Distilled Bert Uncased with a max sequence length of 128 with 64 batches and 10 epochs is 83.6% on the test data with learning rate to 5e-6. 

07/10/2023 22:50 (Bernardo) - Accuracy on Distilled Bert Cased with a max sequence length of 128 with 64 batches and 6 epochs is 82.7% on the test data with a learning rate to 5e-6.

07/10/2023 00:00 (Bernardo) - Accuracy on Bert Based Uncased with a max sequence length of 128 with 32 batches and 6 epochs is 82.6% on the test data with a learning rate to 5e-6.

08/10/2023 16:00 (Bernardo) - Accuracy on Bert Based Uncased with a max sequence length of 128 with 32 batches and 6 epochs is 83% on the test data with a learning rate to 2e-6.

08/10/2023 16:30 (Bernardo) - Accuracy on Bert Based Uncased with a max sequence length of 128 with 32 batches and 12 epochs is 82.1% on the test data with a learning rate to 5e-7.

Going back do DistilBert Uncased form the Best Model Parameters (83.6% Accuracy)

08/10/2023 17:00 (Bernardo) - Accuracy on Distilled Bert Uncased with a max sequence length of 128 with 64 batches and 10 epochs is 82.7% on the test data with learning rate to 5e-6. Adding a Dropout Layer with 10% dropout and a fully connected pre classification linear layer.

08/10/2023 17:30 (Bernardo) - Accuracy on Distilled Bert Uncased with a max sequence length of 128 with 64 batches and 14 epochs is 82.9% on the test data with learning rate to 2e-6. Adding a Dropout Layer with 30% dropout and a fully connected pre classification linear layer.

09/10/2023 15:10 (Everyone) - Accuracy on Distilled Bert Uncased with a max sequence length of 256 with 64 batches and 10 epochs is 83.7% on the test data with learning rate to 5e-6. Overfits around epoch 4.
---> NEW BEST!

09/10/2023 23:00 (Bernardo) - Accuracy on Distilled Bert Uncased with a max sequence length of 256 with 64 batches and 20 epochs is 83.1% on the test data with learning rate using linear scheduler with 1 to 0.3 ratio in 10 epochs. Overfits around epoch 10.
