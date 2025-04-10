SELECT messages.* FROM messages, chunks 
WHERE messages.id = chunks.id_message AND chunks.text = $1;
