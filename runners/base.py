"""Runner interface. run.py talks only to this, never to model SDKs."""

class BaseRunner:
    name = "base"

    def predict(self, state, questions):
        """Return {"answers": {qid: answer}} in laya answer shape. Must not mutate inputs."""
        raise NotImplementedError

    def info(self):
        """Metadata recorded into the result file."""
        return {"runner": self.name}
