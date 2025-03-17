-- Query to list all documents with a summary count at the end
-- This will return all document information followed by a summary row

WITH doc_info AS (
  SELECT 
    id, 
    title, 
    author, 
    filepath, 
    created_at,
    LENGTH(COALESCE(filepath, '')) > 0 AS has_filepath
  FROM 
    documents
  ORDER BY 
    title ASC
)
SELECT 
  id, 
  title, 
  author, 
  filepath, 
  created_at,
  'DOCUMENT' AS type
FROM 
  doc_info

UNION ALL

SELECT 
  NULL AS id,
  'TOTAL DOCUMENTS: ' || COUNT(*) AS title,
  'VALID DOCUMENTS: ' || SUM(CASE WHEN has_filepath THEN 1 ELSE 0 END) AS author,
  'INVALID DOCUMENTS: ' || SUM(CASE WHEN NOT has_filepath THEN 1 ELSE 0 END) AS filepath,
  NULL AS created_at,
  'SUMMARY' AS type
FROM 
  doc_info;