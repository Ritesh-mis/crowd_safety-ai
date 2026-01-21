def log(explanation):
    with open("alerts.log", "a") as f:
        f.write(explanation + "\n\n")
