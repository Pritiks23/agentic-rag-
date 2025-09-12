class LastmileClient:
    def __init__(self, api_token):
        self.api_token = api_token

    def log_evaluation(self, query, answer):
        print(f"[LastMile] Logging evaluation for query length {len(query)} and answer length {len(answer)}")
