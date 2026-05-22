# 测试逻辑

## 1. 测试目标

验证这份初版代码至少满足三类基本正确性：

1. 接口输入输出符合预期。
2. 上传后的文档能被解析并进入可检索状态。
3. 问答接口在有数据和无数据两种情况下都能返回合理结果。

## 2. 当前已落地测试

### `test_upload_document_and_chat_round_trip`

- 类型：集成测试
- 用例目标：验证上传到问答的完整链路
- 检查点：
  - 上传接口返回 200
  - 返回 `document_id`、`task_id`
  - 后台处理后文档状态变为 `processed`
  - `/chat` 可以命中刚上传的内容
  - 返回结果包含来源文件名

### `test_upload_rejects_unsupported_file_type`

- 类型：接口校验测试
- 用例目标：验证不支持的文件类型会被拒绝
- 检查点：
  - 返回 400
  - 错误信息包含 `Unsupported file type`

### `test_chat_returns_404_when_kb_has_no_processed_chunks`

- 类型：异常路径测试
- 用例目标：验证空知识库不会返回伪造答案
- 检查点：
  - 返回 404
  - 错误信息明确说明没有可用知识

## 3. 建议补充测试

后续如果继续往正式版本推进，建议继续补以下测试：

1. Markdown 图片引用抽取测试。
2. PDF 解析测试，区分纯文本 PDF 和复杂版面 PDF。
3. 队列消费测试，验证多文档连续入队时的状态变化。
4. 检索排序测试，比较不同 chunk 对同一问题的得分顺序。
5. 接入 MinerU 后的回归测试，对比 `pdfplumber` 和 `MinerU` 的 chunk 质量差异。