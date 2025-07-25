from flask import Flask, request, render_template_string
import docker
import threading

app = Flask(__name__)
client = docker.from_env()

html_form = """
<!doctype html>
<title>Docker Login and Push</title>
<h2>Enter Docker Hub Credentials</h2>
<form method=post>
  Docker Username: <input type=text name=username><br>
  Docker Password: <input type=password name=password><br>
  <input type=submit value=Submit>
</form>
<pre>{{output}}</pre>
"""

output_log = ""

@app.route("/", methods=["GET", "POST"])
def index():
    global output_log
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]

        output_log = "Logging in...\n"
        try:
            client.login(username=username, password=password)
            output_log += "Login successful.\n"

            output_log += "Building Docker image...\n"
            image, build_logs = client.images.build(path=".", tag=f"{username}/icecream-api:latest")
            for chunk in build_logs:
                if 'stream' in chunk:
                    output_log += chunk['stream']

            output_log += "\nPushing Docker image...\n"
            push_logs = client.images.push(f"{username}/icecream-api:latest", stream=True, decode=True)
            for chunk in push_logs:
                if 'status' in chunk:
                    output_log += chunk['status'] + "\n"
                elif 'error' in chunk:
                    output_log += "Error: " + chunk['error'] + "\n"

            output_log += "Done."
        except Exception as e:
            output_log += f"Error: {e}"

    return render_template_string(html_form, output=output_log)


def run_app():
    app.run(port=5000)

if __name__ == "__main__":
    print("Starting server on http://localhost:5000 ...")
    run_app()
