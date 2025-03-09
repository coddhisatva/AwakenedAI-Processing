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
    created_at DESC
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

-- Alternative query to get just the document counts
SELECT 
  COUNT(*) as total_documents,
  COUNT(CASE WHEN filepath IS NOT NULL AND filepath != '' THEN 1 END) as valid_documents,
  COUNT(CASE WHEN filepath IS NULL OR filepath = '' THEN 1 END) as invalid_documents
FROM 
  documents;

-- Query to see the most recent documents added
SELECT 
  id, 
  title, 
  author, 
  filepath, 
  created_at
FROM 
  documents
ORDER BY 
  created_at DESC
LIMIT 50; 