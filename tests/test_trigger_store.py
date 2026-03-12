"""Tests for TriggerStore — SQLite persistence for triggers."""

import time

import pytest

from atlas.store.trigger_store import TriggerStore
from atlas.triggers.models import TriggerDefinition


@pytest.fixture
async def store(tmp_path):
    s = TriggerStore(str(tmp_path / "test.db"))
    await s.init()
    yield s
    await s.close()


def _make_trigger(**kwargs) -> TriggerDefinition:
    defaults = dict(
        trigger_type="cron",
        cron_expr="*/5 * * * *",
        agent_name="echo",
    )
    defaults.update(kwargs)
    return TriggerDefinition(**defaults)


class TestTriggerStoreCRUD:
    async def test_save_and_get(self, store):
        t = _make_trigger(name="test")
        await store.save(t)
        got = await store.get(t.id)
        assert got is not None
        assert got.id == t.id
        assert got.name == "test"
        assert got.trigger_type == "cron"
        assert got.agent_name == "echo"

    async def test_get_nonexistent(self, store):
        assert await store.get("nope") is None

    async def test_update(self, store):
        t = _make_trigger(name="original")
        await store.save(t)
        t.name = "updated"
        t.fire_count = 5
        await store.save(t)
        got = await store.get(t.id)
        assert got.name == "updated"
        assert got.fire_count == 5

    async def test_delete(self, store):
        t = _make_trigger()
        await store.save(t)
        assert await store.delete(t.id) is True
        assert await store.get(t.id) is None

    async def test_delete_nonexistent(self, store):
        assert await store.delete("nope") is False


class TestTriggerStoreList:
    async def test_list_all(self, store):
        for i in range(3):
            await store.save(_make_trigger(name=f"t{i}"))
        result = await store.list()
        assert len(result) == 3

    async def test_list_by_type(self, store):
        await store.save(_make_trigger(name="a", trigger_type="cron", cron_expr="* * * * *"))
        await store.save(_make_trigger(name="b", trigger_type="webhook"))
        result = await store.list(trigger_type="cron")
        assert len(result) == 1
        assert result[0].name == "a"

    async def test_list_by_enabled(self, store):
        t1 = _make_trigger(name="on", enabled=True)
        t2 = _make_trigger(name="off", enabled=False)
        await store.save(t1)
        await store.save(t2)
        on = await store.list(enabled=True)
        off = await store.list(enabled=False)
        assert len(on) == 1
        assert on[0].name == "on"
        assert len(off) == 1
        assert off[0].name == "off"

    async def test_list_limit(self, store):
        for i in range(5):
            await store.save(_make_trigger(name=f"t{i}"))
        result = await store.list(limit=2)
        assert len(result) == 2


class TestTriggerStoreListDue:
    async def test_due_triggers(self, store):
        now = time.time()
        t1 = _make_trigger(name="due")
        t1.next_fire = now - 10
        t2 = _make_trigger(name="future")
        t2.next_fire = now + 1000
        await store.save(t1)
        await store.save(t2)
        due = await store.list_due(before=now)
        assert len(due) == 1
        assert due[0].name == "due"

    async def test_excludes_webhooks(self, store):
        now = time.time()
        t = _make_trigger(trigger_type="webhook")
        t.next_fire = now - 10
        await store.save(t)
        due = await store.list_due(before=now)
        assert len(due) == 0

    async def test_excludes_disabled(self, store):
        now = time.time()
        t = _make_trigger(enabled=False)
        t.next_fire = now - 10
        await store.save(t)
        due = await store.list_due(before=now)
        assert len(due) == 0

    async def test_excludes_zero_next_fire(self, store):
        t = _make_trigger()
        t.next_fire = 0.0
        await store.save(t)
        due = await store.list_due(before=time.time())
        assert len(due) == 0


class TestTriggerStoreSerialization:
    async def test_input_data_roundtrip(self, store):
        t = _make_trigger(input_data={"key": "value", "nested": {"a": 1}})
        await store.save(t)
        got = await store.get(t.id)
        assert got.input_data == {"key": "value", "nested": {"a": 1}}

    async def test_metadata_roundtrip(self, store):
        t = _make_trigger(metadata={"tag": "test"})
        await store.save(t)
        got = await store.get(t.id)
        assert got.metadata == {"tag": "test"}

    async def test_enabled_bool_roundtrip(self, store):
        t = _make_trigger(enabled=False)
        await store.save(t)
        got = await store.get(t.id)
        assert got.enabled is False


class TestTriggerStoreNotInitialized:
    async def test_save_raises(self):
        store = TriggerStore("nonexistent.db")
        with pytest.raises(RuntimeError, match="not initialized"):
            await store.save(_make_trigger())

    async def test_get_raises(self):
        store = TriggerStore("nonexistent.db")
        with pytest.raises(RuntimeError, match="not initialized"):
            await store.get("x")

    async def test_list_raises(self):
        store = TriggerStore("nonexistent.db")
        with pytest.raises(RuntimeError, match="not initialized"):
            await store.list()

    async def test_list_due_raises(self):
        store = TriggerStore("nonexistent.db")
        with pytest.raises(RuntimeError, match="not initialized"):
            await store.list_due(before=0)

    async def test_delete_raises(self):
        store = TriggerStore("nonexistent.db")
        with pytest.raises(RuntimeError, match="not initialized"):
            await store.delete("x")
