CREATE TABLE users (
    id VARCHAR(255) PRIMARY KEY,
    search_default VARCHAR(255),
    release_only BOOLEAN,
    name VARCHAR(255)
);

CREATE TABLE prompts (
    id VARCHAR(255) PRIMARY KEY,
    text TEXT,
    timestamp TIMESTAMPTZ NOT NULL,
    id_user VARCHAR(255) REFERENCES users (id) 
);

CREATE TABLE answers (
    id VARCHAR(255) PRIMARY KEY,
    text TEXT,
    timestamp TIMESTAMPTZ NOT NULL,
    feedback BOOLEAN,
    id_prompt VARCHAR(255) REFERENCES prompts (id)
);

CREATE TABLE chats (
    id VARCHAR(255) PRIMARY KEY,
    name VARCHAR(255)
);

CREATE TABLE messages (
    id VARCHAR(255) PRIMARY KEY,
    text TEXT,
    timestamp TIMESTAMPTZ NOT NULL,
    sender VARCHAR(255),
    release_date VARCHAR(255),
    id_chat VARCHAR(255)  REFERENCES chats (id)
);

CREATE TABLE chunks (
    id SERIAL PRIMARY KEY,
    text TEXT,
    embedding float[],
    id_message VARCHAR(255) REFERENCES messages (id),
    release_date VARCHAR(255)
);

