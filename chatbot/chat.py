import openai


def main():

    openai.api_key = open("ID/my_api_key.txt").read()

    # read the model ID
    model_ID_file = "ID/my_fine-tuned_model_ID.txt"
    with open(model_ID_file, 'r') as f:
        model_ID = f.read()
    f.close()

    retrieve_response = openai.FineTune.retrieve(model_ID)
    fine_tuned_model = retrieve_response.fine_tuned_model

    if fine_tuned_model is None:
        print("Fine tuned model not ready yet!")

    model_file = "ID/my_fine-tuned_model.txt"
    with open(model_file, 'w') as f:
        f.write(fine_tuned_model)
    f.close()

    return 0

if __name__ == '__main__':
    main()