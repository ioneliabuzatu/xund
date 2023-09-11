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
    :return chunks: list of the chunked words
    """
    def store_entity(start_pos_chunk, end_chunk, entity) -> bool:
        before_start = start_pos_chunk <= entity.position[0] < end_chunk
        before_end = start_pos_chunk < entity.position[1] <= end_chunk
        store = before_start and before_end
        return store

    num_words = len(words)
    chunks = []
    sliding_window = 0

    while sliding_window < num_words:
        start_chunk = sliding_window
        end_chunk = min(sliding_window + max_chunk_size, num_words)

        # check if the end is cutting through an entity
        for entity in entities:
            entity_start, entity_end = entity.position
            if start_chunk <= entity_start < end_chunk < entity_end:
                end_chunk = entity_start
            elif end_chunk >= entity_end > start_chunk > entity_start:
                end_chunk = entity_start

        collected_chunk_entities = [entity for entity in entities if store_entity(start_chunk, end_chunk, entity)]
        chunks.append(Chunk((start_chunk, end_chunk), collected_chunk_entities))
        sliding_window = end_chunk

    return chunks


def test_example():
    # test sample
    words = ["The", "patient", "reports", "fever", "and", "cough", "."]
    entities = [Entity("symptom", (3, 4)), Entity("symptom", (5, 6))]
    max_chunk_size = 2

    # find chunks
    chunks = chunkify(words, entities, max_chunk_size)
    
    # tests
    assert len(chunks) == 4
    assert chunks[1].entities == [entities[0]] and chunks[2].entities == [entities[1]]
    assert not chunks[0].entities and not chunks[3].entities

    for chunk in chunks:
        start, end = chunk.position
        size = end - start
        assert 0 < size <= max_chunk_size