"""

    Program to create a Chatbot using Natural Language Processing (NLP) techniques

    Based on a Naive Bayes Classifier trained on labeled data

    Responses are generated based on pre-defined intent-response pairs

    Created by Cai Smith on 01.05.23

"""
import nltk
import numpy as np
import re


def import_data(file):
    """
    Input: file (string) - path to input file

    Output: data (list) - text and intent list

    Description: This function reads the input file and extracts the text
    and intent information from it. The extracted information is stored
    in a list of dictionaries format, where each dictionary contains the
    text and intent.

    How it works: The function opens the file, loops through each line,
    strips any whitespaces, extracts text and intent information using
    regex, and appends it to the 'data' list. The list of dictionaries
    is returned as output.

    """
    data = []
    # read the data from file
    with open(file, 'r') as file:
        # create the list
        for line in file:
            line = line.strip()
            if not line:
                continue
            text, intent = re.findall(r'^([^,]+),\s*(.+)$', line)[0]
            data.append({"text": text, "intent": intent})

    # print(data)
    return data


def import_reponses(file):
    """

      Input: file \(string\) \- path to input file
      Output: responses \(dictionary\) \- intent and its associated texts

      Description:
      This function read the input file and extract the intent and the text after that it store
      it in the Dictionary in key value pairs, key is the intent and text is the feature of the intent

      How it works:
      it open the file in the read mode and check it ine by line, After that it will extract the intent
      from each of the line using regular expression and if intent is not present in the regular expression
      it will create a new key value pair with an empty list as its value and finally
      it will print an return dictionary

  """

    # read the data from file
    with open(file) as f:
        data = f.read().splitlines()

    # create the dictionary
    responses = {}
    for line in data:
        text, intent = re.findall(r'^(.*),\s*(.*)$', line)[0]
        if intent not in responses:
            responses[intent] = []
        responses[intent].append(text)

    # print the dictionary
    print(responses)
    return responses


def preprocess(text):
    """
        Input: text (string)

        Output: preprocessed_text (string)

        Description: This function preprocess the text by tokenizing it, stemming it, and removing stop words.

        How it works: The function uses the Natural Language Toolkit (NLTK) to tokenize the text, stem the words, and remove stop words. The preprocessed text is returned as output.

        """
    # tokenize the text
    tokenizer = nltk.RegexpTokenizer(r'\w+')
    tokens = tokenizer.tokenize(text)

    # stem the words
    stemmer = nltk.PorterStemmer()
    stemmed_tokens = [stemmer.stem(token) for token in tokens]

    # remove stop words
    stop_words = nltk.corpus.stopwords.words('english')
    preprocessed_text = ' '.join([word for word in stemmed_tokens if word not in stop_words])

    return preprocessed_text


def extract_features(text):
    """
        Input: text (string)

        Output: features (dictionary)

        Description: This function extracts features from the text using a bag-of-words approach.

        How it works: The function uses the Natural Language Toolkit (NLTK) to tokenize the text,
         stem the words, and remove stop words.
         The words are then counted and the frequencies are used as features.

        """
    # This function extracts entities from the text.

    # Print a message to indicate that the function is extracting entities.
    print("extract_entities: text=", text)

    # Get the set of words in the text after preprocessing.
    text_words = set(preprocess(text))

    # Create a dictionary to store the features.
    features = {}

    # For each word in the list of word features,
    for word in word_features:
        # Add a feature to the dictionary indicating whether the word appears in the text.
        features[word] = (word in text_words)

    # Return the features dictionary.
    return features


def evaluate_intent_classifier(classifier):
    """
       Input: classifier (NaiveBayesClassifier)


       Description: This function evaluates the performance of the intent classifier on a test dataset.

       How it works: The function first loads the test dataset, then calculates the confusion matrix,
       precision, recall, and f-measure for each class.
       The function finally prints the confusion matrix, precision, recall, and f-measure for all classes.

       """
    # This code evaluates the performance of the intent classifier on a test dataset.

    # Load the test dataset.
    test_data = import_data("test.txt")

    # Get the list of intents in the test dataset.
    labels = list(set(data['intent'] for data in test_data))

    # Sort the list of intents.
    labels.sort()

    # Print the list of intents.
    print(labels)

    # Get the number of classes.
    num_classes = len(labels)

    # Create a confusion matrix.
    confusion_matrix = [[0] * num_classes for _ in range(num_classes)]

    # Evaluate the classifier on the test dataset.
    for data in test_data:
        # Get the sentence and actual intent.
        sentence = data['text']
        actual_intent = data['intent']

        # Extract features from the sentence.
        features = extract_features(sentence)

        # Predict the intent of the sentence.
        predicted_intent = classifier.classify(features)

        # Get the true label.
        true_label = labels.index(actual_intent)

        # Get the predicted label.
        predicted_label = labels.index(predicted_intent)

        # Update the confusion matrix.
        confusion_matrix[true_label][predicted_label] += 1

    # Calculate the precision, recall, and f-measure for each class.
    precision = [0] * num_classes
    recall = [0] * num_classes
    f_measure = [0] * num_classes

    for i in range(num_classes):

        # Get the true positives.
        true_positives = confusion_matrix[i][i]

        # Get the false positives.
        false_positives = sum(confusion_matrix[j][i] for j in range(num_classes) if j != i)

        # Get the false negatives.
        false_negatives = sum(confusion_matrix[i][j] for j in range(num_classes) if j != i)

        # Calculate the precision.
        if (true_positives + false_positives) > 0:
            precision[i] = true_positives / (true_positives + false_positives)

        # Calculate the recall.
        if (true_positives + false_negatives) > 0:
            recall[i] = true_positives / (true_positives + false_negatives)

        # Calculate the f-measure.
        if (precision[i] + recall[i]) > 0:
            f_measure[i] = 2 * precision[i] * recall[i] / (precision[i] + recall[i])

    # Calculate the average precision, recall, and f-measure.
    avg_precision = sum(precision) / num_classes
    avg_recall = sum(recall) / num_classes
    avg_f_measure = sum(f_measure) / num_classes

    # Print the confusion matrix.
    print("confusion_matrix :", confusion_matrix)

    # Print the precision.
    print("precision", avg_precision)

    # Print the recall.
    print("recall", avg_recall)

    # Print the f-measure.
    print("f_measure", avg_f_measure)


def train_intent_classifier(data):
    """
        Input: data (list of dictionaries)

        Output: classifier (NaiveBayesClassifier)

        Description: This function trains an intent classifier using a Naive Bayes classifier.

        How it works: The function first extracts the text and intent from
        each dictionary in the data, then creates a frequency distribution of all words in the text.
        The function then selects the 1000 most common words and creates a feature vector for each sentence
        in the data. The feature vector for a sentence contains a binary value for each of the 1000 words,
         indicating whether the word appears in the sentence or not.
         The function finally trains a Naive Bayes classifier on the feature vectors and returns the
         classifier.

        """
    # Get the text and intent from each dictionary in the data.
    texts = [d["text"] for d in data]
    intents = [d["intent"] for d in data]
    # Create a frequency distribution of all words in the text
    all_words = nltk.FreqDist([stem for text in texts for stem in preprocess(text)])

    global word_features
    # Select the 1000 most common words.

    word_features = list(all_words.keys())[:1000]
    # Create feature vectors for each sentence.
    featuresets = [(extract_features(text), intent) for text, intent in zip(texts, intents)]
    # Train a Naive Bayes classifier on the feature vectors
    classifier = nltk.NaiveBayesClassifier.train(featuresets)
    return classifier


def generate_response(text):
    """
        Input: text (string)

        Output: response (string)

        Description: This function generates a response to the user's input using the trained intent
        classifier.

        How it works: The function first preprocess the text, then extracts the features from the text,
        then uses the intent classifier to predict the intent of the user's input, then selects a random
        response from the list of responses associated with the predicted intent,
        and finally returns the response.

        """

    # Preprocess the text.
    stems = preprocess(text)
    # Extract features from the text.
    features = extract_features(stems)
    # Predict the intent of the user's input.
    intent = classifier.classify(features)
    # Import the list of responses associated with the predicted intent.
    responses = import_reponses("response.txt")
    # Select a random response from the list of responses.
    response = np.random.choice(responses[intent])
    return response


def chat():
    """
       Description: This function starts a chat with the user.

       How it works: The function prints a greeting message,
       then enters a loop where it prompts the user for input and responds to the user's input.
       The function exits the loop when the user enters "exit".
       """
    print("Hi, I'm ChatBot. How can I help you today?")

    while True:
        # Get the user's input.
        user_input = input("> ")
        if user_input.lower() == "exit":
            break
        # Print the response.
        response = generate_response(user_input)
        print(response)


# train the data set
train_data = import_data("data.txt")
classifier = train_intent_classifier(train_data)
evaluate_intent_classifier(classifier)

chat()

