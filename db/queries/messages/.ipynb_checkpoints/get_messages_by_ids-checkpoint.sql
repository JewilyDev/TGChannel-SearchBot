SELECT messages.* FROM messages 
WHERE messages.id = ANY($1);
