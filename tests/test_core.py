from core import Chunk, KnowledgeBase, _split_words, validated_citations


def test_split_words_overlap_and_short_input():
    assert list(_split_words("uno dos tres cuatro cinco", size=3, overlap=1)) == [
        "uno dos tres",
        "tres cuatro cinco",
    ]
    assert list(_split_words("", size=3, overlap=1)) == []


def test_search_prefers_relevant_chunk():
    chunks = [
        Chunk("P1-F1", 1, "A decision tree classifier learns from labeled training examples."),
        Chunk("P2-F1", 2, "A neural network contains layers of connected artificial neurons."),
    ]
    kb = KnowledgeBase(chunks)
    result = kb.search("decision tree classifier training examples", top_k=2)
    assert result
    assert result[0][0].source_id == "P1-F1"
    assert result[0][1] > 0


def test_citation_validator_keeps_only_retrieved_ids():
    answer = "Está respaldado [S1], pero esto no [S8]."
    assert validated_citations(answer, {"S1"}) == (
        "Está respaldado [S1], pero esto no [referencia no verificada]."
    )
