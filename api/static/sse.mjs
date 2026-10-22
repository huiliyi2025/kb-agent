// 按行解析 SSE，支持任意网络分片、CRLF、多行 data 和注释心跳。
export class SSEParser {
  constructor(onEvent) {
    this.onEvent = onEvent;
    this.buffer = "";
    this.event = "message";
    this.data = [];
  }
  feed(text) {
    this.buffer += text;
    let newline;
    while ((newline = this.buffer.indexOf("\n")) >= 0) {
      const line = this.buffer.slice(0, newline).replace(/\r$/, "");
      this.buffer = this.buffer.slice(newline + 1);
      if (line === "") {
        if (this.data.length) this.onEvent(this.event, JSON.parse(this.data.join("\n")));
        this.event = "message";
        this.data = [];
      } else if (!line.startsWith(":")) {
        const colon = line.indexOf(":");
        const field = colon < 0 ? line : line.slice(0, colon);
        const value = colon < 0 ? "" : line.slice(colon + 1).replace(/^ /, "");
        if (field === "event") this.event = value;
        if (field === "data") this.data.push(value);
      }
    }
  }
}
