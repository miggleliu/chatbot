# imports
import openai
import pandas as pd
import numpy as np
import json
import sys
import warnings
from flask import Flask, request, jsonify, render_template
from ast import literal_eval


# models
EMBEDDING_MODEL = "text-embedding-ada-002"
GPT_MODEL = "gpt-3.5-turbo"
DF_FILE = "output/chatbot_embedded_reviews.csv"


def main():
    update_data = False
    if len(sys.argv) != 2:
        warnings.warn("Argument 'update_data' is expected if you need the data to be updated, otherwise by default the data remains the same!", UserWarning)
        update_data = False
    else:
        if sys.argv[1] == 'update_data':
            update_data = True
        else:
            warnings.warn("Meaningless Argument! Argument 'update_data' is expected if you need the data to be updated, otherwise by default the data remains the same!", UserWarning)
            update_data = False

    openai.api_key = open("ID/my_api_key.txt").read().strip()

    # create/read df which stores information of embeddings
    if update_data:
        data_list = create_data_list("data/original_data.json")
        df = pd.DataFrame(columns=['Context', 'Embedding', 'Similarity'])
        for data in data_list:
            category, command, function = data[0], data[1], data[2]
            context = "{category}: {command}; Function: {function}".format({"category": category, "command": command, "function": function})
            embedding = get_embedding(context)
            new_row = [context, embedding, -1]
            df.loc[len(df)] = new_row
        df.to_csv(DF_FILE, index=False)
    else:
        df = pd.read_csv(DF_FILE)
        df['Embedding'] = df['Embedding'].apply(literal_eval)

    # runs app
    app = Flask(__name__)

    @app.route("/", methods=["GET"])
    def index():
        return render_template("index.html")

    @app.route("/get_response", methods=["POST"])
    def get_response():
        user_input = request.form["user_input"]

        # read query and search for the most relevant contexts
        top_relevant_df = search_reviews(df, user_input, n=3)
        message = query_message(user_input, top_relevant_df)

        print("Message:\n", message)

        messages = [
            {"role": "system", "content": "You answer questions about different linux commands."},
            {"role": "user", "content": message},
        ]
        response = openai.ChatCompletion.create(
            model=GPT_MODEL,
            messages=messages,
            temperature=0
        )

        bot_response = response["choices"][0]["message"]["content"].strip()
        return jsonify({"bot_response": bot_response})

    app.run(debug=True)

    return 0


def query_message(
    query: str,
    df: pd.DataFrame,
) -> str:
    """Return a message for GPT, with relevant source texts pulled from a dataframe."""
    introduction = 'Use the below given knowledge to answer the subsequent question. If the answer cannot be found in the articles, write "I could not find an answer."'
    question = f"\n\nQuestion: {query}"
    message = introduction
    for context in df['Context']:
        message += f"\n{context}"
    return message + question


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


def get_embedding(text, model=EMBEDDING_MODEL):
    text = text.replace("\n", " ")
    return openai.Embedding.create(input=[text], model=model)['data'][0]['embedding']


def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def search_reviews(df, query, n=3):
    embedding = get_embedding(query)
    df['Similarity'] = df.Embedding.apply(lambda x: cosine_similarity(x, embedding))
    res = df.sort_values('Similarity', ascending=False).head(n)
    print(res['Similarity'])
    return res


if __name__ == '__main__':
    main()
