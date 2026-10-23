#!/usr/bin/env python3
"""Edit the blocked-word list, which lives in a private object on COS and not
in the (public) repository.

  python scripts/policy_tool.py pull   # COS -> ./content_policy.json (git-ignored)
  # edit content_policy.json
  python scripts/policy_tool.py push   # validate, then ./content_policy.json -> COS (private)

The next pipeline run re-checks every live episode against the new list.
Needs TENCENT_COS_SECRET_ID / TENCENT_COS_SECRET_KEY (e.g. in .env).
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import koto_lib as K  # noqa: E402


def main():
    if len(sys.argv) != 2 or sys.argv[1] not in ("pull", "push"):
        sys.exit(__doc__)
    storage = K.Storage()
    if sys.argv[1] == "pull":
        data = storage.get_private_json(K.POLICY_KEY)
        if data is None:
            sys.exit(f"{K.POLICY_KEY} 不存在")
        K.POLICY_FILE.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"已下载到 {K.POLICY_FILE}（{sum(len(v) for v in data['blocked_terms'].values())} 个词）")
        return
    data = json.loads(K.POLICY_FILE.read_text(encoding="utf-8"))
    policy = K.load_policy(K.POLICY_FILE)  # raises on a malformed file
    if not policy["terms"]:
        sys.exit("词表为空，拒绝上传")
    storage.put_json(K.POLICY_KEY, data, cache_control="no-cache", private=True)
    print(f"已上传 {len(policy['terms'])} 个词到 {K.POLICY_KEY}（私有）。下一次运行会按新词表复查全部单集。")


if __name__ == "__main__":
    main()

