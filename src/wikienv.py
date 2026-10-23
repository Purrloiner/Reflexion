import wikipedia

wikipedia.set_lang("en")


class WikiEnv:

    def __init__(self):
        self.page = ""
        self.sentences = []

        self.lookup_keyword = ""
        self.lookup_results = []
        self.lookup_index = 0

    def search(self, query):

        try:
            results = wikipedia.search(query)

            if len(results) == 0:
                return f"Could not find [{query}]"

            page = wikipedia.page(
                query,
                auto_suggest=False
            )

            self.page = page.content

            self.sentences = [
                s.strip()
                for s in self.page.replace("\n", " ").split(". ")
                if s.strip()
            ]

            self.lookup_keyword = ""
            self.lookup_results = []
            self.lookup_index = 0

            paragraphs = [
                p.strip()
                for p in self.page.split("\n\n")
                if p.strip()
            ]

            if paragraphs:
                return paragraphs[0]

            return "\n".join(self.sentences[:3])

        except wikipedia.exceptions.DisambiguationError as e:
            return (
                f"Could not find [{query}]. "
                f"Similar: {e.options[:10]}"
            )

        except Exception:
            try:
                results = wikipedia.search(query)
                return (
                    f"Could not find [{query}]. "
                    f"Similar: {results[:10]}"
                )
            except Exception:
                return f"Could not find [{query}]"

    def lookup(self, keyword):

        keyword = keyword.strip().lower()

        if keyword != self.lookup_keyword:

            self.lookup_keyword = keyword

            self.lookup_results = [
                sentence
                for sentence in self.sentences
                if keyword in sentence.lower()
            ]

            self.lookup_index = 0

        if len(self.lookup_results) == 0:
            return f"No sentence containing [{keyword}]"

        if self.lookup_index >= len(self.lookup_results):
            return f"No more results for [{keyword}]"

        sentence = self.lookup_results[self.lookup_index]

        result_number = self.lookup_index + 1
        total_results = len(self.lookup_results)

        self.lookup_index += 1

        return (
            f"(Result {result_number} / {total_results}) "
            f"{sentence}"
        )

    def finish(self, answer):
        return answer