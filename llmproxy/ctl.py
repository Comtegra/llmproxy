import argparse
import asyncio
import datetime
import hashlib
import secrets
import sys

from . import config
from .db import DatabaseError, get_db


def isodatetime(s):
    if s == "now":
        return datetime.datetime.now(datetime.timezone.utc)

    return datetime.datetime.fromisoformat(s).astimezone(datetime.timezone.utc)


parser = argparse.ArgumentParser("llmproxyctl",
    description="Manage an instance of LLM Proxy")
parser.add_argument("-c", "--config", type=config.load, default="",
    help="path to configuration file")
subparsers = parser.add_subparsers(required=True)

# Command: user
parser_user = subparsers.add_parser("user", help="API key management")
subparsers_user = parser_user.add_subparsers(required=True)


# Command: user create

async def command_user_create(args):
    db = await get_db(uri=args.config["db"]["uri"])

    secret = secrets.token_urlsafe(64)
    digest = hashlib.sha256(secret.encode()).hexdigest()

    try:
        await db.user_create(digest, args.expires, args.comment)
    except DatabaseError as e:
        print("Database error:", e, file=sys.stderr)
        sys.exit(1)

    await db.close()

    print("User created", file=sys.stderr)
    print("Expires:", args.expires or "never", file=sys.stderr)
    print("Comment:", args.comment or "-", file=sys.stderr)
    print("Hash:", digest[:12])
    print("Plain API key:", end=" ", file=sys.stderr)
    sys.stderr.flush()
    print(secret)

parser_user_create = subparsers_user.add_parser("create")
parser_user_create.add_argument("-e", "--expires", type=isodatetime,
    help="expiration time in ISO 8601 format")
parser_user_create.add_argument("-t", "--comment",
    help="arbitrary text associated with the key")
parser_user_create.set_defaults(func=command_user_create)


if __name__ == "__main__":
    args = parser.parse_args()
    asyncio.run(args.func(args))
