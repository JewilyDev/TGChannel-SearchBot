UPDATE users SET search_default = ($1)
WHERE id = ($2);
