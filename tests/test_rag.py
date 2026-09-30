import io
import json
import unittest
from contextlib import redirect_stderr, redirect_stdout
from itertools import pairwise
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace as NS
from unittest.mock import Mock, call, patch

from pypdf import PdfWriter

import documents
import main
from documents import Chunk, load_documents, split_text


class DocumentTests(unittest.TestCase):
    def test_split_overlap_coverage_and_bounds(self):
        text = "".join(chr(i) for i in range(33, 127))
        parts = split_text(text, size=17, overlap=5)
        self.assertTrue(all(0 < len(part) <= 17 for part in parts))
        for left, right in pairwise(parts):
            self.assertEqual(left[-5:], right[:5])
        self.assertEqual(parts[0] + "".join(p[5:] for p in parts[1:]), text)
        self.assertEqual("".join(split_text(text, size=17, overlap=0)), text)
        self.assertEqual(split_text("  alpha\n beta\t gamma  "), ["alpha beta gamma"])
        self.assertEqual(split_text(" \n\t "), [])
        self.assertEqual(split_text("abc", size=1, overlap=0), list("abc"))
        self.assertEqual(split_text("a b", size=1, overlap=0), ["a", "b"])
        self.assertEqual(
            split_text("alpha beta gamma delta", size=12, overlap=0),
            ["alpha beta", "gamma delta"],
        )

    def test_split_invalid_size_and_overlap(self):
        for size, overlap in [(0, 0), (-1, 0), (8, -1), (8, 8), (8, 9)]:
            with (
                self.subTest(size=size, overlap=overlap),
                self.assertRaises(ValueError),
            ):
                split_text("text", size=size, overlap=overlap)

    def test_load_utf8_metadata_and_unsupported_files(self):
        with TemporaryDirectory() as tmp:
            folder = Path(tmp)
            (folder / "a.TXT").write_text("caf\u00e9\nmenu", encoding="utf-8-sig")
            (folder / "b.md").write_text("# R\u00e9sum\u00e9", encoding="utf-8")
            (folder / "ignored.csv").write_bytes(b"\xff")
            (folder / "nested.txt").mkdir()
            chunks, warnings = load_documents(folder)
        self.assertEqual(warnings, [])
        self.assertEqual(
            chunks,
            [
                Chunk("caf\u00e9 menu", "a.TXT", 1),
                Chunk("# R\u00e9sum\u00e9", "b.md", 1),
            ],
        )

    def test_missing_and_empty_folder(self):
        with TemporaryDirectory() as tmp:
            self.assertEqual(load_documents(Path(tmp)), ([], []))
            with self.assertRaises(ValueError):
                load_documents(Path(tmp) / "missing")

    def test_file_and_collection_limits(self):
        with TemporaryDirectory() as tmp:
            folder = Path(tmp)
            (folder / "a.txt").write_text("first document", encoding="utf-8")
            (folder / "b.txt").write_text("second document", encoding="utf-8")
            with patch.object(documents, "MAX_FILE_BYTES", 1):
                chunks, warnings = load_documents(folder)
            self.assertEqual(chunks, [])
            self.assertEqual(len(warnings), 2)
            with (
                patch.object(documents, "MAX_CHUNKS", 1),
                self.assertRaises(ValueError),
            ):
                load_documents(folder)

    def test_corrupt_blank_and_encrypted_pdf_warnings(self):
        with TemporaryDirectory() as tmp:
            folder = Path(tmp)
            (folder / "corrupt.pdf").write_bytes(b"not a PDF")
            for name in ["blank.pdf", "encrypted.pdf"]:
                writer = PdfWriter()
                writer.add_blank_page(width=72, height=72)
                if name == "encrypted.pdf":
                    writer.encrypt("secret")
                with (folder / name).open("wb") as output:
                    writer.write(output)
            with redirect_stderr(io.StringIO()):
                chunks, warnings = load_documents(folder)
        self.assertEqual(chunks, [])
        self.assertEqual(len(warnings), 3)
        for name in ["corrupt.pdf", "blank.pdf", "encrypted.pdf"]:
            self.assertTrue(any(name in warning for warning in warnings), name)
        self.assertTrue(any("Okunabilir metin yok" in w for w in warnings))
        self.assertTrue(any("Sifreli PDF" in w for w in warnings))

    def test_pdf_page_metadata_numbering_and_blank_page_warning(self):
        pages = [
            Mock(extract_text=Mock(return_value=text))
            for text in ["A" * 900, None, "last page"]
        ]
        with TemporaryDirectory() as tmp:
            folder = Path(tmp)
            (folder / "report.pdf").write_bytes(b"mocked PDF")
            with patch.object(
                documents, "PdfReader", return_value=NS(is_encrypted=False, pages=pages)
            ):
                chunks, warnings = load_documents(folder)
        self.assertEqual([c.page for c in chunks], [1, 1, 3])
        self.assertEqual([c.number for c in chunks], [1, 2, 3])
        self.assertTrue(all(c.source == "report.pdf" for c in chunks))
        self.assertEqual([c.text for c in chunks], ["A" * 800, "A" * 220, "last page"])
        self.assertEqual(len(warnings), 1)
        self.assertIn("report.pdf: 1 bos/metinsiz sayfa", warnings[0])


class RetrievalTests(unittest.TestCase):
    def test_cosine(self):
        self.assertAlmostEqual(main.cosine_similarity([3, 4], [6, 8]), 1)
        self.assertAlmostEqual(main.cosine_similarity([1, 0], [-1, 0]), -1)
        self.assertEqual(main.cosine_similarity([1, 0], [0, 1]), 0)
        self.assertEqual(main.cosine_similarity([0, 0], [1, 2]), 0)
        with self.assertRaises(ValueError):
            main.cosine_similarity([1, 2], [1])

    def test_ranking_threshold_and_top_k(self):
        vectors = [[0, 1], [3, 4], [1, 0], [-1, 0]]
        self.assertEqual(
            main.find_relevant([1, 0], vectors, top_k=2, min_score=0),
            [(2, 1.0), (1, 0.6)],
        )
        self.assertEqual(main.find_relevant([1, 0], vectors, min_score=1), [(2, 1.0)])
        self.assertEqual(main.find_relevant([1, 0], [[0, 1]], min_score=0.1), [])
        self.assertEqual(main.find_relevant([1, 0], []), [])
        for top_k, min_score in [(0, 0), (1, -1.1), (1, 1.1)]:
            with (
                self.subTest(top_k=top_k, min_score=min_score),
                self.assertRaises(ValueError),
            ):
                main.find_relevant([1, 0], vectors, top_k, min_score)

    def test_embed_batches_and_restores_index_order(self):
        chunks = [Chunk(str(i), "notes.txt", i + 1) for i in range(10)]
        client = Mock()
        client.generate_embeddings.side_effect = lambda texts: NS(
            data=[
                NS(index=i, embedding=[int(texts[i])])
                for i in reversed(range(len(texts)))
            ]
        )
        with redirect_stdout(io.StringIO()):
            result = main.embed_chunks(client, chunks)
        self.assertEqual(result, [[i] for i in range(10)])
        self.assertEqual(
            client.generate_embeddings.call_args_list,
            [call([str(i) for i in range(8)]), call(["8", "9"])],
        )

    def test_embed_rejects_invalid_indices(self):
        chunks = [Chunk("a", "notes.txt", 1), Chunk("b", "notes.txt", 2)]
        for indices in [[], [0], [0, 0], [0, 2], [-1, 0], [0, 1, 2]]:
            client = Mock()
            client.generate_embeddings.return_value = NS(
                data=[NS(index=i, embedding=[1]) for i in indices]
            )
            with self.subTest(indices=indices), self.assertRaises(ValueError):
                main.embed_chunks(client, chunks)

    def test_build_messages_contains_only_retrieved_sources(self):
        chunks = [
            Chunk("excluded secret", "private.txt", 1),
            Chunk("PDF evidence", "report.pdf", 4, 7),
            Chunk("text evidence", "notes.md", 2),
        ]
        messages = main.build_messages("What happened?", chunks, [(2, 0.9), (1, 0.8)])
        self.assertEqual([m["role"] for m in messages], ["system", "user"])
        self.assertNotIn("excluded secret", json.dumps(messages))
        self.assertNotIn("private.txt", json.dumps(messages))
        self.assertEqual(
            json.loads(messages[1]["content"]),
            {
                "question": "What happened?",
                "context": [
                    {
                        "source_id": 1,
                        "file": "notes.md",
                        "page": None,
                        "chunk": 2,
                        "text": "text evidence",
                    },
                    {
                        "source_id": 2,
                        "file": "report.pdf",
                        "page": 7,
                        "chunk": 4,
                        "text": "PDF evidence",
                    },
                ],
            },
        )

    def test_answer_abstains_without_chat(self):
        embedding, chat = Mock(), Mock()
        embedding.generate_embedding.return_value = NS(data=[NS(embedding=[1, 0])])
        with redirect_stdout(io.StringIO()) as output:
            main.answer_question(
                "question",
                [Chunk("text", "a.txt", 1)],
                [[0, 1]],
                embedding,
                chat,
                min_score=0.5,
            )
        embedding.generate_embedding.assert_called_once_with("question")
        self.assertEqual(chat.mock_calls, [])
        self.assertIn("modelden yanit istenmedi", output.getvalue())

    def test_answer_handles_empty_stream(self):
        for stream in [
            [],
            [
                NS(choices=[]),
                NS(choices=[NS(delta=NS(content=None))]),
                NS(choices=[NS(delta=NS(content=""))]),
            ],
        ]:
            with self.subTest(stream=stream):
                embedding, chat = Mock(), Mock()
                embedding.generate_embedding.return_value = NS(
                    data=[NS(embedding=[1, 0])]
                )
                chat.complete_streaming_chat.return_value = iter(stream)
                with redirect_stdout(io.StringIO()) as output:
                    main.answer_question(
                        "question",
                        [Chunk("text", "a.pdf", 1, 2)],
                        [[1, 0]],
                        embedding,
                        chat,
                    )
                chat.complete_streaming_chat.assert_called_once()
                self.assertIn("Model bos yanit dondurdu", output.getvalue())
                self.assertIn("a.pdf, sayfa 2", output.getvalue())

    def test_inspect_does_not_initialize_foundry(self):
        with TemporaryDirectory() as tmp:
            (Path(tmp) / "notes.txt").write_text("preview evidence", encoding="utf-8")
            with (
                patch.object(main.FoundryLocalManager, "initialize") as initialize,
                redirect_stdout(io.StringIO()) as output,
            ):
                status = main.main(["--docs", tmp, "--inspect"])
            initialize.assert_not_called()
        self.assertEqual(status, 0)
        self.assertIn("notes.txt", output.getvalue())
        self.assertIn("preview evidence", output.getvalue())
        self.assertIn("model baslatilmadi", output.getvalue())

    def test_terminal_flow_and_cleanup(self):
        chunks = [Chunk("Python is supported.", "sdk.md", 1)]
        answer = NS(choices=[NS(delta=NS(content="Python [1]"))])
        for mode in ["success", "retry", "interrupt"]:
            with self.subTest(mode=mode):
                manager, embedding_model, chat_model = Mock(), Mock(), Mock()
                for model in (embedding_model, chat_model):
                    model.variants = [
                        Mock(info=NS(runtime=NS(device_type="GPU"))),
                        Mock(info=NS(runtime=NS(device_type="CPU"))),
                    ]
                manager.catalog.get_model.side_effect = [embedding_model, chat_model]
                client = embedding_model.get_embedding_client.return_value
                client.generate_embeddings.return_value = NS(
                    data=[NS(index=0, embedding=[1, 0])]
                )
                client.generate_embedding.return_value = NS(data=[NS(embedding=[1, 0])])
                stream = chat_model.get_chat_client.return_value.complete_streaming_chat
                stream.return_value = [answer]
                inputs = ["", "Which language?", "quit"]
                if mode == "retry":
                    stream.side_effect = [
                        main.FoundryLocalException("test failure"),
                        [answer],
                    ]
                    inputs = ["first question", "second question", "quit"]
                elif mode == "interrupt":
                    inputs = [KeyboardInterrupt()]
                with (
                    patch.object(main, "load_documents", return_value=(chunks, [])),
                    patch.object(main.FoundryLocalManager, "initialize"),
                    patch.object(main.FoundryLocalManager, "instance", manager),
                    patch("builtins.input", side_effect=inputs),
                    redirect_stdout(io.StringIO()) as output,
                ):
                    if mode == "interrupt":
                        with self.assertRaises(KeyboardInterrupt):
                            main.main([])
                    else:
                        self.assertEqual(main.main([]), 0)
                        self.assertIn("Python [1]", output.getvalue())
                        self.assertIn("sdk.md, parca 1", output.getvalue())
                embedding_model.unload.assert_called_once()
                chat_model.unload.assert_called_once()
                for model in (embedding_model, chat_model):
                    model.select_variant.assert_called_once_with(model.variants[1])
                if mode == "retry":
                    self.assertIn("Soru islenemedi", output.getvalue())
                    self.assertEqual(stream.call_count, 2)


if __name__ == "__main__":
    unittest.main()
