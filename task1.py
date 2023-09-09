from time import perf_counter
from typing import Tuple, Sequence, Collection, List, NamedTuple


class Entity(NamedTuple):
    """Representation of an entity"""
    # type of entity
    name: str
    # start and end index of entity in some word sequence (interval end is exclusive)
    position: Tuple[int, int]


class Chunk(NamedTuple):
    """Chunk representation"""
    # start and end index of chunk in some word sequence (interval end is exclusive)
    position: Tuple[int, int]
    # entities inside this chunk
    entities: Collection[Entity]


def chunkify(words: Sequence[str], entities: Collection[Entity], max_chunk_size: int) -> List[Chunk]:
    """Split words into chunks with a maximum size

    :param words: a sequence of words
    :param entities: entities associated with words
    :param max_chunk_size: maximum number of words a chunk is allowed to contain
    """
    chunks = []
    i = 0

    while i < len(words):
        start = i
        end = min(i + max_chunk_size, len(words))

        # Check if the end is cutting through an entity
        for entity in entities:
            entity_start, entity_end = entity.position
            if start <= entity_start < end < entity_end:
                end = entity_start
            elif end >= entity_end > start > entity_start:
                end = entity_start

        # Collect entities for the current chunk
        chunk_entities = [entity for entity in entities if start <= entity.position[0] < end and start < entity.position[1] <= end]

        chunks.append(Chunk((start, end), chunk_entities))
        i = end

    return chunks


def test_example():
    # test sample
    words = ["The", "patient", "reports", "fever", "and", "cough", "."]
    entities = [Entity("symptom", (3, 4)), Entity("symptom", (5, 6))]
    max_chunk_size = 2

    # find chunks
    chunks = chunkify(words, entities, max_chunk_size)
    
    print(chunks)

    _chunks = [
        Chunk((0, 2), []),
        Chunk((2, 4), [Entity("symptom", (3, 4))]),
        Chunk((4, 6), [Entity("symptom", (5, 6))]),
        Chunk((6, 7), [])
    ]

    # tests
    assert len(chunks) == 4
    assert chunks[1].entities == [entities[0]] and chunks[2].entities == [entities[1]]
    assert not chunks[0].entities and not chunks[3].entities

    start = perf_counter()
    for chunk in chunks:
        start, end = chunk.position
        size = end - start
        assert 0 < size <= max_chunk_size