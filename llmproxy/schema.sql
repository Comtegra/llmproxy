CREATE TABLE IF NOT EXISTS event_oneoff (
	id INTEGER PRIMARY KEY,
	created TEXT,
	api_key TEXT,
	product TEXT,
	quantity INTEGER,
	rid TEXT
);

CREATE TABLE IF NOT EXISTS api_key (
	id TEXT PRIMARY KEY,
	secret BLOB,
	type TEXT,
	expires TEXT,
	comment TEXT
);
