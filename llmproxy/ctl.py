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


# Command: user list

async def command_user_list(args):
    db = await get_db(uri=args.config["db"]["uri"])

    fmt = "%(secret)-12.12s  %(expires)-20s  %(status)-7s  %(comment)s"
    print(fmt % {"secret": "Hash", "expires": "Expires", "status": "Status",
        "comment": "Comment"})
    print(fmt % {"secret": "-" * 12, "expires": "-" * 20, "status": "-" * 7,
        "comment": "-------"})

    try:
        users = await db.user_list(args.hash, include_expired=True)
    except DatabaseError as e:
        print("Database error:", e, file=sys.stderr)
        sys.exit(1)

    for user in users:
        if user["expires"] is not None:
            try:
                user["expires"] = datetime.datetime.fromisoformat(user["expires"])\
                    .astimezone().strftime("%Y-%m-%d %H:%M:%S")
            except (ValueError, TypeError):
                user["expires"] = "???"
        else:
            user["expires"] = "-"

        print(fmt % user)

    await db.close()

parser_user_list = subparsers_user.add_parser("list")
parser_user_list.add_argument("hash", nargs="?", default="",
    help="hash prefix to use for filtering the list")
parser_user_list.set_defaults(func=command_user_list)


# Command: user update

async def command_user_update(args):
    if args.expires is None and args.comment is None:
        print("No fields to update. Specify -e or -t.", file=sys.stderr)
        sys.exit(1)

    update = {}

    if args.expires == "":
        update["expires"] = None
    elif args.expires is not None:
        try:
            update["expires"] = isodatetime(args.expires)
        except ValueError as e:
            print("Failed to parse date:", e, file=sys.stderr)
            sys.exit(1)

    if args.comment is not None:
        update["comment"] = args.comment

    db = await get_db(uri=args.config["db"]["uri"])

    try:
        users = await db.user_list(args.hash, include_expired=True)
        if not users:
            print("User not found.", file=sys.stderr)
            await db.close()
            sys.exit(1)

        if len(users) > 1:
            print("More than one user found. Specify more of the hash prefix.",
                file=sys.stderr)
            await db.close()
            sys.exit(1)

        await db.user_update(users[0], **update)
    except DatabaseError as e:
        print("Database error:", e, file=sys.stderr)
        await db.close()
        sys.exit(1)

    await db.close()

    print("User updated", file=sys.stderr)

parser_user_update = subparsers_user.add_parser("update")
parser_user_update.add_argument("-e", "--expires")
parser_user_update.add_argument("-t", "--comment")
parser_user_update.add_argument("hash")
parser_user_update.set_defaults(func=command_user_update)


if __name__ == "__main__":
    args = parser.parse_args()
    asyncio.run(args.func(args))
