# NLI_project
A conversational agent trained on Restaurant and Hotel domains (Erasmus Mundus EMAI project)

## Team

| **Name / Surname** | **GitHub** |
| :---: | :---: | :---: |
| `Camille` | [![name](https://github.com/b-rbmp/NexxGate/blob/main/docs/logos/github.png)](https://github.com/CLendering) |
| `Bernardo` | [![name](https://github.com/b-rbmp/NexxGate/blob/main/docs/logos/github.png)](https://github.com/b-rbmp) |
| `Roberta` | [![name](https://github.com/b-rbmp/NexxGate/blob/main/docs/logos/github.png)](https://github.com/RobCTs) |
| `Nazanin` | [![name](https://github.com/b-rbmp/NexxGate/blob/main/docs/logos/github.png)](https://github.com/Naominickels) |


## Implementation of the core functionalities of a CA
  • Domain identification/ Dialog act prediction

  • Content extraction from User utterances (semantic frame slot filling)

  • Agent move prediction

Data set: https://huggingface.co/datasets/multi_woz_v2i (i=2,3,4)


## Approach:
  - **Dialog Act Prediction Model**: A classification model that takes user utterance as input and predicts the dialog act. This could be an LSTM or Transformer-based model. (Classification)
    - Classifies the type of action the user is attempting to perform (e.g., asking a question, making a statement).    
  - **Semantic Slot Filling Model**: This could be a sequence-to-sequence model or a named entity recognition (NER) model. (Classification)
    - Extracts key pieces of information (e.g., date, location) from the user's utterance.
  - **Agent Move Prediction Model**: Use reinforcement learning or rule-based methods to decide the agent's next move based on the dialog history and current state. (Reinforcement Learning)
    - Determines the next action the conversational agent should take (e.g., provide information, ask for clarification).     
