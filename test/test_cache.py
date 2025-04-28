# test_cache.py
import numpy as np
from src.cache import ItemCache  # Import the class from your cache.py file


def test_add_and_remove_specific_item():
    cache = ItemCache(min_counter=1, max_counter=3)
    specific_item = np.array([1, 2, 3, 4, 5])
    cache.add_item_with_specific_counter(specific_item, 2)  # Adding with a counter of 2

    # Assuming the update_counters method removes the item when its counter reaches 0
    cache.update_counters()  # Decrease counter by 1
    assert len(cache.cache) == 1, "Item should not be removed yet"

    removed_items = cache.update_counters()  # This should remove the item
    assert len(cache.cache) == 0, "Item should be removed"
    assert len(removed_items) == 1, "One item should be in the removed items list"
    assert (removed_items[0] == specific_item).all(
    ), "The removed item should match the specific item"


def test_add_random_item():
    cache = ItemCache(min_counter=1, max_counter=3)
    cache.add_item_with_random_counter("item")
    assert len(cache.cache) == 1, "One item should be added"
    # Additional assertions can check the properties of the added item
