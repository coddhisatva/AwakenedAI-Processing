-- SQL to completely reset the database
-- IMPORTANT: This will permanently delete ALL data from your database!

-- 1. First delete all chunks (due to foreign key constraints)
DELETE FROM chunks;

-- 2. Then delete all documents
DELETE FROM documents;

-- 3. Verify the deletion
SELECT COUNT(*) FROM chunks;
SELECT COUNT(*) FROM documents; 