from flask import Flask, render_template, request
from models import analyze_sentiments, plot_grouped_bar_chart, load_models
from cryptography.fernet import Fernet

app = Flask(__name__)
models = load_models()

# Encryption Key
key = Fernet.generate_key()
cipher_suite = Fernet(key)

@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        text = request.form.get("text")
        action = request.form.get("action")

        if action == "Encrypt":
            encrypted_text = cipher_suite.encrypt(text.encode()).decode()
            return render_template("index.html", encrypted_text=encrypted_text)

        elif action == "Analyze Sentiment":
            results = analyze_sentiments(models, text)
            graph_path = "static/sentiment_graph.png"
            plot_grouped_bar_chart(results, output_path=graph_path)
            return render_template("index.html", analysis_results=results, graph_path=graph_path)

    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
