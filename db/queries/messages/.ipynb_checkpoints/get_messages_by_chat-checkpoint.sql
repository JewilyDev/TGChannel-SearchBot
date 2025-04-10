SELECT messages.* FROM messages, chats 
WHERE messages.id_chat = chats.id AND chats.id = ($1);
