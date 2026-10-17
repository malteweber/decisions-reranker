from dataclasses import dataclass

from openai import OpenAI
from openai.types.decision import AnswerAnswerResourcePredicate
from openai.types.decision_create_params import QuestionQuestionParamPredicate

DEFAULT_INSTRUCTION = "Is this snippet relevant to answer the given query: {document}"


@dataclass
class Document:
    id: str
    text: str

@dataclass
class ScoredDocument:
    id: str
    text: str
    score: float


def _validate_instruction(instruction: str) -> str:
    if "{document}" not in instruction:
        raise ValueError("instruction must contain the placeholder '{document}'")
    try:
        instruction.format(document="")
    except (KeyError, IndexError) as error:
        raise ValueError(f"instruction contains an unknown placeholder: {error}") from error
    return instruction


class DecisionsReranker:
    def __init__(
        self,
        client: OpenAI | None = None,
        model: str = "gpt-6-luna",
        instruction: str = DEFAULT_INSTRUCTION,
        threshold: float = 0.5
    ) -> None:
        self._client = client or OpenAI()
        self._model = model
        self._instruction = _validate_instruction(instruction)
        self._threshold = threshold

    def filter(
        self,
        query: str,
        documents: list[Document],
        threshold: float | None = None,
        instruction: str | None = None,
    ) -> list[ScoredDocument]:
        cutoff = self._threshold if threshold is None else threshold
        filtered = [
            ScoredDocument(document.id, document.text, score)
            for document, score in self._score(query, documents, instruction)
            if score > cutoff
        ]
        filtered_sorted = sorted(filtered, key=lambda d: d.score, reverse=True)
        return filtered_sorted

    def rerank(
        self,
        query: str,
        documents: list[Document],
        instruction: str | None = None,
    ) -> list[ScoredDocument]:
        scored = sorted(self._score(query, documents, instruction), key=lambda item: item[1], reverse=True)
        return [ScoredDocument(document.id, document.text, score) for document, score in scored]

    def _score(
        self,
        query: str,
        documents: list[Document],
        instruction: str | None,
    ) -> list[tuple[Document, float]]:
        template = self._instruction if instruction is None else _validate_instruction(instruction)
        decision = self._client.decisions.create(
            model=self._model,
            input=query,
            questions=[
                QuestionQuestionParamPredicate(
                    type="predicate",
                    name=document.id,
                    instructions=template.format(document=document.text),
                )
                for document in documents
            ],
        )
        return [
            (document, answer.probability if isinstance(answer, AnswerAnswerResourcePredicate) else 0.0)
            for document, answer in zip(documents, decision.answers, strict=True)
        ]
