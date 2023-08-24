import openai
import json
import ast
from sklearn.model_selection import train_test_split
import os
import time


def main():
    openai.api_key = open("ID/my_api_key.txt").read().strip()

    data_list = create_data_list("data/original_data.json")
    training_question_list = []
    testing_question_list = []

    for data in data_list:
        qna_string = get_questions(data)
        if qna_string == -1:
            continue
        try:
            qna_list = extract_qa(qna_string)
            if len(qna_list) == 1:
                training_question_list += qna_list
            else:
                training_question_list += qna_list[:-1]
                testing_question_list.append(qna_list[-1])
            print(training_question_list)
            print(testing_question_list)
        except:
            continue

    model_ID = train(training_question_list, testing_question_list)
    #model_ID = create_finetune_model(model='curie', batch_size=8, n_epochs=20, training_file="data/training_data.jsonl", validation_file="data/testing_data.jsonl")
    model_ID_file = "ID/my_fine-tuned_model_ID.txt"
    with open(model_ID_file, "w") as f:
        f.write(model_ID)
    f.close()

    return 0


def create_data_list(original_json_file):
    f = open(original_json_file)
    data = json.load(f)
    data_list = []
    for title in data:
        for command in data[title]:
            description = data[title][command]
            data_list.append([title, command, description])
    f.close()
    print(data_list)
    return data_list


def get_questions(data):
    category, command, function = data[0], data[1], data[2]
    context = f"{category}: {command}\nFunction: {function}"
    content = f"Please write at least five question-answer pairs based on the text below, in the format of a nested python list. Each inner list should contain one question and one answer.\n\n{context}"
    example_context = "command: p4 add\nFunction: to add new files"
    example_content = f"Example:\nPlease write at least five question-answer pairs based on the text below, in the format of a nested python list. Each inner list should contain one question and one answer.\n\n{example_context}"

    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are good at generating question-answer pairs based on a given text."},
                {"role": "user", "content": example_content},
                {"role": "assistant", "content": "qa_list = [\n['What is the p4 command to add a new file?', 'p4 add'],\n['What is the purpose of p4 add?', 'to add a new file'],\n['When do you use p4 add?', 'when you want to add a new file'],\n['What command should you use when you want to add a new file?', 'p4 add'],\n['What is the function of p4 add?','to add new files']\n]"},
                {"role": "user", "content": content}
            ]
        )
    except openai.error.ServiceUnavailableError:
        return -1

    msg = response['choices'][0]['message']['content']

    print("This is the response:\n")
    print(msg)
    return msg


def extract_qa(data_string):
    if '->' in data_string:
        data_string = data_string.replace('->', 'implies')
    if '\n' in data_string:
        data_string = data_string.replace('\n', ' ')

    # Extract the content inside the square brackets using literal_eval
    qa_pairs = ast.literal_eval(data_string.split('=')[-1])
    print(qa_pairs)

    return qa_pairs


# train model, returns the ID of the trained model
def train(train_df, test_df):
    print(train_df)
    print(test_df)
    with open("data/training_data.jsonl", "w") as f:
        for qa_pair in train_df:
            prompt_completion_pair = {"prompt": qa_pair[0] + " ->", "completion": " " + qa_pair[1] + "\n"}
            json.dump(prompt_completion_pair, f)
            f.write("\n")
    f.close()
    with open("data/testing_data.jsonl", "w") as f:
        for qa_pair in test_df:
            prompt_completion_pair = {"prompt": qa_pair[0] + " ->", "completion": " " + qa_pair[1] + "\n"}
            json.dump(prompt_completion_pair, f)
            f.write("\n")
    f.close()

    fine_tuned_model_id = create_finetune_model(model='curie', batch_size=len(train_df), n_epochs=20, training_file="data/training_data.jsonl", validation_file="data/testing_data.jsonl")
    return fine_tuned_model_id


def create_file(file_name):
    upload_response = openai.File.create(
        file=open(file_name, "rb"),
        purpose='fine-tune'
    )
    file_id = upload_response.id
    return file_id


def create_finetune_model(model, batch_size, n_epochs, training_file, validation_file):
    fine_tune_response = openai.FineTune.create(
        model=model,
        batch_size=batch_size,
        n_epochs=n_epochs,
        training_file=create_file(training_file),
        validation_file=create_file(validation_file)
    )

    job_id = fine_tune_response["id"]
    print(f"Fine_tuning job created with ID: {job_id}")

    return job_id


if __name__ == "__main__":
    main()




