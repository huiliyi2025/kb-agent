import test from "node:test";
import assert from "node:assert/strict";
import { SSEParser } from "../api/static/sse.mjs";

test("SSE survives every possible network split including CRLF and Chinese UTF-8", () => {
  const wire = ': heartbeat\r\nevent: token\r\ndata: {"text":"你好"}\r\n\r\nevent: done\ndata: {\ndata: "answer":"完成"}\n\n';
  const bytes = new TextEncoder().encode(wire);
  for (let cut = 0; cut <= bytes.length; cut++) {
    const result = [];
    const parser = new SSEParser((event, data) => result.push([event, data]));
    const decoder = new TextDecoder();
    parser.feed(decoder.decode(bytes.slice(0, cut), { stream: true }));
    parser.feed(decoder.decode(bytes.slice(cut)));
    assert.deepEqual(result, [["token", { text: "你好" }], ["done", { answer: "完成" }]]);
  }
});

test("incomplete event is not published as a finished answer", () => {
  const result = [];
  const parser = new SSEParser((...args) => result.push(args));
  parser.feed('event: done\ndata: {"answer":"truncated"}\n');
  assert.equal(result.length, 0);
});
