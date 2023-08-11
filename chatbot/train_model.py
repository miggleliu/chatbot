import openai
import json
import ast
from sklearn.model_selection import train_test_split
import os
import time


def main():
    # openai.api_key = open("ID/my_api_key.txt").read()
    data_list = create_data_list("data/original_data.json")
    question_list = []

    '''
    i = 0
    for data in data_list:
        qna_string = get_questions(data)
        print(type(qna_string))
        i += 1
        qna_list = extract_qa(qna_string)
        question_list += qna_list

        if i == 15:
            break
    '''
    question_list = [['What is the syntax for removing a workspace in p4?', 'p4w remove <workspace directory>'], ['Why should we use "p4w remove" instead of "rm -rf" to remove a workspace?', 'p4w remove is the proper command for removing workspaces in p4, while "rm -rf" is a Unix/Linux command to forcefully remove directories and files.'], ['What is the purpose of p4w sync_all command?', 'To sync all files and components'], ['What flag can be added to bypass sanity check and skips verification of files?', '-bsc'], ['What is the command to show the description of a changelist?', 'p4 describe <CL>'], ['What is the purpose of p4 describe?', 'to show the description of a changelist'], ['What is the command to edit a file?', 'p4 edit <path of file>'], ['What is the purpose of the p4 edit command?', 'to edit a file'], ['What is the syntax for adding a new file in p4?', 'p4 add <path of file>'], ['What does the "p4 add" command do?', 'It adds a new file to the p4 repository'], ['What argument does the "p4 add" command require?', 'The path of the file that needs to be added'], ['What is the purpose of p4 change?', 'To setup a changelist number for your current edited files in workspace'], ['What does the p4 change command do?', 'It creates a new changelist or activates an existing changelist for the current workspace'], ['How can you use p4 change?', 'You can use it to organize your changes and track your progress while working on multiple files'], ['What is the command to submit a changelist?', 'p4 submit -c <CL>'], ['What does the p4 submit command do?', 'It submits the changelist you currently have'], ['What is the purpose of the "lastgood" command?', 'To print out the latest passing CL'], ['What should you do before kicking off dv_check_submit?', 'Sync to the CL printed by "lastgood"'], ['What is the purpose of dv_check_submit command?', 'runs sanity test on CL that is to be submitted'], ['What parameter is used with dv_check_submit command to specify the changelist?', '-c <CL>'], ['What is the purpose of the command "p4_mkwa"?', 'To create a new workspace'], ['What are the required parameters for the command "p4_mkwa"?', '-codeline, -branch, -changelist'], ['What is the use of the parameter "-codeline" in the "p4_mkwa" command?', 'To specify the codeline for the workspace'], ['What does the parameter "-branch" do in the "p4_mkwa" command?', 'It specifies the branch for the workspace'], ['What is the purpose of the parameter "-changelist" in the "p4_mkwa" command?', 'To sync the workspace to a specific changelist']]
    # model_ID = train(question_list)

    model_ID = "ft-OFWz6icLSI4p9WkJWwtq9ThJ"
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
    title, command, description = data[0], data[1], data[2]
    context = f"{title}:\n{command}: {description}"
    content = f"Please write question-answer pairs based on the text below, in the format of a nested python list. Each inner list should contain one question and one answer.\n\nText:{context}"
    example_context = "p4 command:\n p4 add: add new file"
    example_content = f"Please write question-answer pairs based on the text below, in the format of a nested python list. Each inner list should contain one question and one answer.\n\nFor example, Text:{example_context}"

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant who is good at concluding question-answer pairs based on a given text."},
            {"role": "user", "content": example_content},
            {"role": "assistant", "content": "qa_list = [\n['What is the command to add a new file?', 'p4 add'],\n['What is the purpose of p4 add?', 'to add a new file']\n]"},
            {"role": "user", "content": content}
        ]
    )

    msg = response['choices'][0]['message']['content']

    print("This is the response:\n")
    print(msg)
    return msg


def extract_qa(data_string):
    if data_string.contains('->'):
        data_string = data_string.replace('->', 'implies')
    if data_string.contains('\n'):
        data_string = data_string.replace('\n', ' ')

    # Extract the content inside the square brackets using literal_eval
    qa_pairs = ast.literal_eval(data_string.split('=')[-1])
    print(qa_pairs)

    return qa_pairs


# train model, returns the ID of the trained model
def train(df):
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
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

    fine_tuned_model_id = create_finetune_model(model='curie', batch_size=16, training_file="data/training_data.jsonl", validation_file="data/testing_data.jsonl")
    return fine_tuned_model_id


def create_file(file_name):
    upload_response = openai.File.create(
        file=open(file_name, "rb"),
        purpose='fine-tune'
    )
    file_id = upload_response.id
    return file_id


def create_finetune_model(model, batch_size, training_file, validation_file):
    fine_tune_response = openai.FineTune.create(
        model=model,
        batch_size=batch_size,
        training_file=create_file(training_file),
        validation_file=create_file(validation_file)
    )

    job_id = fine_tune_response["id"]
    print(f"Fine_tuning job created with ID: {job_id}")

    return job_id


if __name__ == "__main__":
    main()




