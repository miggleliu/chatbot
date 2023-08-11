from flask import Flask, request, jsonify, render_template
import openai

model_file = "ID/my_fine-tuned_model.txt"
with open(model_file, 'r') as f:
    fine_tuned_model = f.read()
f.close()

if fine_tuned_model is None:
    print("Fine tuned model not ready yet!")
else:
    print(fine_tuned_model)
    app = Flask(__name__)

    openai.api_key = open("ID/my_api_key.txt").read()

    @app.route("/", methods=["GET"])
    def index():
        return render_template("index.html")

    @app.route("/get_response", methods=["POST"])
    def get_response():
        user_input = request.form["user_input"]

        response = openai.Completion.create(
            engine=fine_tuned_model,
            prompt=user_input,
            max_tokens=50
        )

        bot_response = response.choices[0].text.strip()

        return jsonify({"bot_response": bot_response})


if __name__ == "__main__":
    app.run(debug=True)
