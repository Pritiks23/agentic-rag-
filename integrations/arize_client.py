class ArizeClient:
    def __init__(self, api_key, space_key):
        self.api_key = api_key
        self.space_key = space_key

    def log_text(self, query, answer):
        print(f"[Arize] Logging query and answer. Query length: {len(query)}, Answer length: {len(answer)}")
