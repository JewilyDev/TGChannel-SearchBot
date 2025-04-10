INSERT INTO users (id, name, release_only) VALUES ($1, $2, True) 
ON CONFLICT(id) DO NOTHING;
