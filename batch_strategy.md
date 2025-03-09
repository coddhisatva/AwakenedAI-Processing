# RAG Pipeline Batching Strategy

## 1. Supabase Chunk Insertion Batching

### a) Record Size Analysis
Our chunks have the following characteristics:
- Text content: ~3-5KB per chunk (based on semantic chunking)
- Vector embedding: ~6KB (1536 dimensions for ada-002 model)
- Metadata and IDs: ~0.5KB
- **Total per record**: ~9.5-11.5KB

### b) Optimal Batch Size for Our Implementation
Based on our record size analysis and Supabase performance characteristics:

- **Recommended Batch Size**: **200 items** per batch is optimal for our medium-to-large records
- **Adjustment Strategy**: 
  - If experiencing timeouts or high latency: Reduce to 150 items
  - If performance is excellent: Consider testing up to 250 items
- **Implementation Method**: Use the JavaScript/TypeScript Supabase client for better error handling, or PostgreSQL's native batch operations with the `unnest` function

### c) Additional Preparation for Success - Concrete Actions

1. **Indexing Strategy**: 
   - Create specific index on the `documents` table for filepath lookups
   - Add index for document-to-chunk relationships
   - Consider embedding vector index using pgvector's `ivfflat` for large collections

2. **Transaction Management**: 
   - Implement transaction wrapping for each batch insertion
   - Set appropriate transaction isolation level (Read Committed is typically sufficient)
   - Establish transaction timeout parameters appropriate for batch size

3. **Performance Monitoring**: 
   - Implement simple timing metrics to measure insertion rate (chunks/second)
   - Log batch sizes and completion times for performance analysis
   - Create a monitoring dashboard to track progress across all documents

4. **Rate Limiting Mitigation**: 
   - Implement exponential backoff with configurable retry count
   - Add jitter to retry delays to prevent thundering herd problems
   - Set maximum timeout thresholds for individual batches

5. **Query Optimization**: 
   - Execute `EXPLAIN ANALYZE` on batch insertions in test environment
   - Review and optimize query plans for document lookup operations
   - Consider disabling triggers or constraints during bulk operations

6. **Resource Allocation**: 
   - Monitor PostgreSQL connection pool utilization
   - Adjust work_mem parameter temporarily for bulk operations
   - Consider temporary maintenance_work_mem increase for index operations

### d) Testing Approach for Supabase Batching

1. **Database-Specific Testing**:
   - Test chunk insertion with controlled batch sizes (50, 100, 200, 300)
   - Measure insertion time and success rate for each batch size
   - Monitor Supabase connection pool utilization
   - Capture and analyze database errors at different batch sizes

2. **Transaction Boundary Testing**:
   - Test partial batch failures and rollback behavior
   - Verify data consistency after failed transactions
   - Measure transaction commit time at different batch sizes
   - Test system behavior when approaching Supabase timeout limits

3. **Performance Measurement**:
   - Record chunks-per-second insertion rate across multiple batch sizes
   - Measure network latency between application and Supabase
   - Test with and without vector indexing to measure impact
   - Analyze performance with varying database load conditions

4. **Resilience Testing**:
   - Simulate network interruptions during batch operations
   - Test retry logic with exponential backoff
   - Verify idempotent operations work correctly on retry
   - Validate fallback to smaller batch sizes on repeated failures

5. **Scaling Tests**:
   - Measure performance degradation with increasing database size
   - Test concurrent batch insertions from multiple processes
   - Analyze index performance as vector store grows
   - Determine optimal maintenance window timing for vacuum operations

## 2. Embedding API Token Limits

### a) OpenAI Embedding API Characteristics

The text-embedding-ada-002 model has the following specifications:
- **Maximum Context Length**: 8,191 tokens per input
- **Vector Dimensions**: 1,536 dimensions per embedding
- **API Rate Limits**: 
  - Tier 1 (free trial users): 150 RPM (requests per minute)
  - Tier 2 (pay-as-you-go users): 3,000 RPM
- **Pricing**: $0.0001 per 1,000 tokens (10× cheaper than completing the same text)
- **Request Format**: Can send multiple input texts as an array for batch processing
- **Response Time**: Generally faster for batch requests than individual ones
- **Token Calculation**: Each input text's tokens count separately toward the 8,191 limit

### b) Optimal Batch Size Strategy

Based on OpenAI's embedding API characteristics and our processing requirements:

- **Recommended Batch Size**: **500 items** per embedding API call is optimal
- **Adjustment Strategy**:
  - If experiencing rate limit errors: Reduce to 200-300 items
  - If performance is excellent: Consider testing up to 1,000 items
  - For very long chunks: Use smaller batches (100-200) to stay under token limits
- **Implementation Method**: Use the OpenAI Python SDK's ability to process arrays of inputs

### c) Additional Preparation for Success - Concrete Actions

1. **Token Management**:
   - Implement token counting for each chunk before batching
   - Group chunks into batches that won't exceed the 8,191 token limit
   - Sort chunks by token length to optimize batch composition
   - Implement automatic batch splitting when token limits would be exceeded

2. **Rate Limit Handling**:
   - Implement exponential backoff for rate limit errors
   - Track requests per minute and self-throttle to stay under limits
   - Add jitter to retry delays to prevent synchronization problems
   - Consider implementing a token bucket algorithm for precise rate control

3. **Error Resilience**:
   - Develop specific handlers for common API errors:
     * Rate limit exceeded errors
     * Token limit exceeded errors
     * Timeout errors
     * Service unavailable errors
   - Log failed chunks for manual review or special handling
   - Implement gradual batch size reduction when encountering errors

4. **Cost Optimization**:
   - Track token usage per batch to monitor and control costs
   - Consider implementing token usage forecasting before starting large jobs
   - Evaluate cost/benefit of different batch sizes (larger batches are more cost-efficient)
   - Develop a budget monitoring system with configurable alerts

5. **Performance Monitoring**:
   - Measure embeddings generated per second at different batch sizes
   - Track API response times to identify optimal batch sizes
   - Monitor memory usage during embedding generation
   - Implement timing metrics to identify bottlenecks

### d) Testing Approach for Embedding API Batching

1. **Token Limit Testing**:
   - Test with chunks of varying sizes to verify token limit handling
   - Intentionally create batches approaching the 8,191 token limit
   - Verify that oversized batches are correctly split
   - Measure performance impact of batch splitting

2. **Rate Limit Testing**:
   - Gradually increase request frequency to identify actual rate limits
   - Test backoff and retry mechanisms with simulated rate limit errors
   - Verify that the system can recover from rate limit errors
   - Benchmark sustainable throughput under different account tiers

3. **Batch Size Optimization**:
   - Compare throughput (embeddings/second) across different batch sizes
   - Test batch sizes from 50 to 1,000 items in 50-item increments
   - Measure cost efficiency (tokens processed per dollar) at each batch size
   - Identify the optimal batch size for our specific use case

4. **Error Handling Verification**:
   - Simulate API errors using a mock server
   - Test system recovery from connection interruptions
   - Verify data consistency after error recovery
   - Ensure no data loss during partial batch failures

5. **Long-Running Test**:
   - Process 1,000+ chunks in a single run
   - Monitor system performance over time
   - Verify consistent embedding quality across the run
   - Identify any degradation in performance or reliability during extended operation

## 3. Evaluating Batching in Earlier Phases

We recommend **against implementing batching** for document extraction and chunking processes. Extraction is I/O bound with limited performance gains, while chunking sees significant memory increases with batch size. Focus optimization efforts on the embedding and storage phases where batching provides clear benefits.

## 4. Integrated Pipeline Testing Framework

This section provides a framework for testing the entire RAG pipeline with all batching components working together.

### a) Small-Scale Validation
- Test the complete pipeline with 1-2 documents (a few hundred chunks)
- Compare end-to-end performance across multiple configuration settings
- Measure and record integrated metrics:
  - Total processing time from document to stored vector
  - Success rate across all pipeline phases
  - Bottleneck identification between phases
  - Memory usage patterns throughout the pipeline

### b) Scale Testing
- Process 5-10 diverse documents through the complete pipeline
- Monitor for cross-phase bottlenecks and error propagation
- Analyze how batch sizes in one phase affect subsequent phases
- Validate memory usage and resource patterns across the complete process

### c) Metrics Collection and Analysis
- End-to-end processing time per document
- Phase-transition efficiency (time between phases)
- Success/failure rates at phase boundaries
- System resource utilization across the complete pipeline
- Cost analysis per document and per chunk
- Performance comparison against baseline (non-batched) approach

### d) Test Data Management
- Create representative document corpus for repeatable testing
- Include documents of varying sizes, formats, and complexity
- Establish performance baselines for comparison
- Create automated test harness for regression testing as implementation evolves

## 5. Risk Mitigation Strategy

### a) Fallback Mechanisms
- Implement automatic batch size reduction when errors occur
- Create single-item insertion fallback for consistently problematic chunks
- Develop special handling for oversized or complex documents

### b) Progress Preservation
- Implement document-level checkpointing
- Create resumable processing capability
- Maintain a detailed processing log for debugging and recovery
- Design idempotent operations to prevent duplicate processing

### c) Phased Implementation
- Separate pipeline into independently executable phases:
  1. Document extraction 
  2. Chunk creation
  3. Embedding generation
  4. Database storage
- Allow each phase to be run and verified separately
- Enable processing to resume from any phase

## 6. Incremental Implementation Approach

### a) Reliability-First Implementation
- Begin with conservative batch sizes (100) to establish baseline
- Focus on completing end-to-end pipeline before optimizing
- Implement comprehensive error handling and logging
- Process documents sequentially before attempting parallelization

### b) Optimization Phase
- Incrementally increase batch sizes based on performance data
- Implement batching at embedding generation first (highest impact)
- Add chunk insertion batching once embedding is stable
- Consider document extraction batching only after other phases are optimized

### c) Scaling Phase
- Introduce parallelization only after batch processing is stable
- Scale up gradually with careful monitoring
- Establish performance baselines at each scale level
- Develop automatic throttling based on system metrics

## 7. Comprehensive Metrics Across Pipeline Phases

After analyzing the existing metrics in the RAG pipeline, we need to extend them to cover all phases of the pipeline with batch-specific metrics:

### a) Current Metrics Assessment
The RAG pipeline already tracks:
- Document-level statistics (total, successful, failed, skipped)
- High-level chunking metrics (total chunks created)
- Processing timings for major phases
- Token limit handling

However, these metrics are **not cumulative** across batches and focus primarily on extraction success rather than processing efficiency across all phases.

### b) Phase-Specific Metrics 

#### 1. Extraction Phase Metrics
- Batch extraction performance (documents per second)
- Format-specific metrics (PDF vs EPUB performance)
- OCR detection and processing statistics
- Memory usage during extraction
- Extraction quality measurements

#### 2. Chunking Phase Metrics
- Semantic chunking throughput (chunks per second)
- Chunk size distribution statistics
- Overlap effectiveness metrics
- Memory usage during batch chunking
- Batch size impact on chunking quality

#### 3. Embedding Phase Metrics
- Embedding generation throughput at different batch sizes
- API rate limit monitoring and avoidance statistics
- Token usage optimization metrics
- Batch embedding cost efficiency
- Error handling and recovery time
- Retry statistics for failed batches

#### 4. Storage Phase Metrics
- Database insertion throughput at different batch sizes
- Transaction success/failure rates
- Network latency to Supabase
- Connection pool utilization
- Batch size optimization for database performance
- Index usage statistics

### c) Cross-Phase Monitoring Types

1. **Batch-Specific Metrics**:
   - Processing time for each batch across all phases
   - Success/failure rates per batch size in each phase
   - Memory consumption patterns per batch size
   - Resource utilization by phase and batch size

2. **Cumulative Processing Metrics**:
   - Running average of items processed per second by phase
   - Cumulative success rates across all batches by phase
   - Long-term performance trends across larger document sets
   - Total pipeline efficiency measurements

3. **Rate Limiting and Recovery Metrics**:
   - API rate limit encounters and resolution time
   - Automatic batch size adjustments based on performance
   - Recovery statistics from batch failures in each phase
   - Self-healing mechanism effectiveness

### d) Implementation Approach
Rather than replacing the existing metrics system, we'll extend it by:

1. Adding a `PhaseMonitor` class for each pipeline phase
2. Extending the existing `Timer` class to support phase-specific batch timing
3. Adding cumulative counters to the existing `stats` dictionary with phase segmentation 
4. Implementing persistence of metrics between pipeline runs
5. Creating a simple dashboard to visualize performance by phase and batch size

This approach will provide comprehensive visibility into all four phases of the pipeline while enabling data-driven optimization of each phase's batch processing strategy. 