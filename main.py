import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import tensorflow as tf
from tensorflow.keras import layers, losses, metrics

# Step 1: Read and tokenize the data
def read_and_tokenize_data(data_paths, labels_paths):
    questions = []
    labels = []
    label_to_index = {}  # Dictionary to map labels to unique numeric indices
    index = 0

    for data_path, label_path in zip(data_paths, labels_paths):
        with open(data_path, 'r', encoding='utf-8') as data_file, \
                open(label_path, 'r', encoding='utf-8') as label_file:
            for question, label in zip(data_file, label_file):
                # Convert text to numeric values using UTF-8 encoding
                question_numeric = [ord(char) for char in question.strip()]
                questions.append(question_numeric)

                # Encode labels into numeric format
                label = label.strip()
                if label not in label_to_index:
                    label_to_index[label] = index
                    index += 1
                labels.append(label_to_index[label])

    return questions, labels

train_data_paths = [
    "Database/en/GettingInfo/Time/Training.txt",
    "Database/en/FormingAnswer/Time/Training.txt"
]
train_labels_paths = [
    "Database/en/GettingInfo/Time/TrainingLabels.txt",
    "Database/en/FormingAnswer/Time/TrainingLabels.txt"
]
validation_data_paths = [
    "Database/en/GettingInfo/Time/Validating.txt",
    "Database/en/FormingAnswer/Time/Validating.txt"
]
validation_labels_paths = [
    "Database/en/GettingInfo/Time/ValidatingLabels.txt",
    "Database/en/FormingAnswer/Time/ValidatingLabels.txt"
]

train_questions, train_labels = read_and_tokenize_data(train_data_paths, train_labels_paths)
validation_questions, validation_labels = read_and_tokenize_data(validation_data_paths, validation_labels_paths)

# Step 2: Pad or truncate sequences to a fixed length
max_sequence_length = 50
train_questions_padded = tf.keras.preprocessing.sequence.pad_sequences(train_questions, maxlen=max_sequence_length)
validation_questions_padded = tf.keras.preprocessing.sequence.pad_sequences(validation_questions, maxlen=max_sequence_length)

# Step 3: Prepare data for training
train_dataset = tf.data.Dataset.from_tensor_slices((train_questions_padded, train_labels))
validation_dataset = tf.data.Dataset.from_tensor_slices((validation_questions_padded, validation_labels))

# Step 4: Define the Model
model = tf.keras.Sequential([
    layers.Embedding(input_dim=100, output_dim=64, input_shape=(max_sequence_length,)),
    layers.GlobalAveragePooling1D(),
    layers.Dense(64, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

# Step 5: Compile the Model
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Step 6: Train the Model
history = model.fit(train_dataset,
                    epochs=100,
                    validation_data=validation_dataset)

# Step 7: Evaluate the Model
loss, accuracy = model.evaluate(validation_dataset)

# Step 8: Save the Model
model.save('my_model')
