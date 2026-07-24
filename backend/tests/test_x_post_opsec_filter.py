from __future__ import annotations

# X_OPSEC_LLM_REWRITE_SPEC の回帰テスト。
#
# 【方針転換(X_OPSEC_LLM_REWRITE_SPEC)】旧 opsec は、層1(候補選定)で
# 単語/正規表現マッチ、層2(配信直前)で正規表現マッチにより actionable を
# 弾いていたが、これが「投稿なし」の目詰まりを起こしていた。現行は:
#   - 層1: 候補選定段階では opsec 除外をしない(素材を残し素通し)。
#   - 層2: 配信直前に LLM 一回の判定＋書き直し(x_opsec_rewrite)へ一本化。
# よって本ファイルは、層1 が素通しになっている(=機材候補が残る)回帰と、
# 層2 の LLM 書き直し経路(test_x_opsec_rewrite.py)を分担する。
#
# filter_private_info() の正規表現自体は、返信経路(reply_approval.py・
# x_reply_generator.py)・デバッグ用途で引き続き使われるため関数として残って
# おり、その actionable 検出の挙動もここで回帰として押さえておく(X 投稿経路
# からはもう呼ばれない)。ネットワーク非依存(DB/LLM に触れない・決定的)。

from app.services.x_post_category_selector import _public_safe_confirm_candidates
from app.services.x_privacy_filter import filter_private_info


# ─── 層1: 候補選定段階では opsec 除外をしない(素通しの回帰) ──────────
def test_layer1_passes_through_all_candidates():
    # 機材・存在レベルも、actionable な具体を含む候補も、候補選定段階では
    # 一切除外しない(actionable の除去は配信直前の LLM 書き直しに委ねる)。
    candidates = [
        {"category": "devices", "key": "home_server", "value": "自宅サーバーでGTX1660を動かしてる"},
        {"category": "environment", "key": "os", "value": "OSはUbuntu Server"},
        {"category": "environment", "key": "expose", "value": "SIM対応ルータで外部公開、ポート開放してる"},
        {"category": "devices", "key": "ip", "value": "IPは 192.168.0.11"},
    ]
    safe = _public_safe_confirm_candidates(candidates)
    # 全件そのまま残る(目詰まり=候補全滅を起こさない)。
    assert [c["key"] for c in safe] == [c["key"] for c in candidates]


def test_layer1_machine_level_candidates_all_kept():
    # 機材レベルばかりでも全部残る(=公開材料が消えて「投稿なし」にならない)。
    candidates = [
        {"category": "environment", "key": "home_server", "value": "自宅でAIサーバー運用してる"},
        {"category": "devices", "key": "gpu", "value": "GTX1660 6GB"},
    ]
    assert len(_public_safe_confirm_candidates(candidates)) == 2


def test_layer1_empty_stays_empty():
    assert _public_safe_confirm_candidates([]) == []


# ─── filter_private_info の actionable 検出(返信経路向けに残る) ────────
# X 投稿経路からは呼ばれなくなったが、reply_approval.py 等で使われ続けるため
# 挙動を回帰として押さえる。
def _blocked(text: str) -> bool:
    safe, _detected = filter_private_info(text)
    return not safe


def test_filter_private_info_blocks_actionable():
    assert _blocked("100.64.1.5 でアクセスできる")            # CGNAT
    assert _blocked("192.168.0.11 が自宅サーバー")            # 既存 IPv4
    assert _blocked("ポート12345を開けて外部公開してる")       # ポート+外部公開
    assert _blocked("port 8080 を公開")
    assert _blocked("api_key=abcd1234efgh")                    # 認証情報(既存)
    assert _blocked("固定IPでDDNS設定した")
    assert _blocked("ポートフォワーディングを設定")
    assert _blocked("SSIDは myhome_wifi")
    assert _blocked("sigmaris.local に繋いでる")               # 内部ホスト名
    assert _blocked("Tailscaleで 100.100.1.2 のノードに繋ぐ")  # 文脈依存+接地あり


def test_filter_private_info_does_not_block_mere_mention():
    assert not _blocked("自宅サーバーでGTX1660を動かしてる")
    assert not _blocked("使ってるOSはUbuntu Serverです")
    assert not _blocked("Tailscaleでリモート開発してる")       # 接地なし
    assert not _blocked("新しいアーキテクチャの設計を考えた")
    assert not _blocked("今日はよく眠れた")
    assert not _blocked("モバイルルータで外に出てる")           # 一般語のみ
