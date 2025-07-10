// utils.ts - With fixed embedding generation

import { pipeline } from '@xenova/transformers';
import { Pinecone } from "@pinecone-database/pinecone";

let _extractor: any = null;

async function getExtractor() {
    if (!_extractor) {
        // Initialize the pipeline with the same model
        _extractor = await pipeline('feature-extraction', 'mixedbread-ai/mxbai-embed-large-v1', {
            quantized: false,
            revision: 'main'  // Specify the model revision
        });
    }
    return _extractor;
}

export async function queryPineconeVectorStore(
    client: Pinecone,
    indexName: string,
    namespace: string,
    query: string,
    topK: number = 5
): Promise<string> {
    try {
        console.log("Generating embedding for query:", query);
        const extractor = await getExtractor();
        
        // FIX: Make sure we're getting a proper embedding
        const output = await extractor(query, { 
            pooling: 'mean',  // Try mean pooling instead of cls
            normalize: true   // Normalize the output vectors
        });
        
        // Convert to regular array (this should be a high-dimensional vector)
        let queryEmbedding;
        
        // Handle different output formats from the extractor
        if (output.data) {
            // Some models return { data: Float32Array }
            queryEmbedding = Array.from(output.data);
        } else if (output.tolist) {
            // Some models have a tolist() method
            queryEmbedding = Array.from(output.tolist());
        } else if (Array.isArray(output)) {
            // Some models return arrays directly
            queryEmbedding = output;
        } else {
            // Last resort: try to access as array or convert to array
            queryEmbedding = Array.from(output);
        }

        console.log("Generated embedding with length:", queryEmbedding.length);
        console.log("Sample values:", queryEmbedding.slice(0, 5));
        
        if (queryEmbedding.length <= 1) {
            throw new Error("Invalid embedding generated: dimension too small");
        }
        
        // Get the index and specify namespace at the index level
        const index = client.Index(indexName);
        const namespaceObj = index.namespace(namespace);
        
        // Use direct vector query without namespace parameter
        const queryResponse = await namespaceObj.query({
            topK: topK,
            vector: queryEmbedding,
            includeMetadata: true
        });
        
        console.log("Query response received:", queryResponse?.matches?.length || 0, "matches");
        console.log("Full query response structure:", JSON.stringify(queryResponse, null, 2));
        
        // Log all fetched chunks with their scores and metadata
        if (queryResponse.matches && queryResponse.matches.length > 0) {
            console.log("=== FETCHED CHUNKS ===");
            queryResponse.matches.forEach((match, index) => {
                console.log(`Chunk ${index + 1}:`);
                // Log all properties of the match object to see what's available
                console.log(`  Full match object:`, JSON.stringify(match, null, 2));
                
                // Try different property names that might contain the score
                const possibleScoreProps = ['_score', 'score', 'similarity', 'distance'];
                const scoreValue = possibleScoreProps.find(prop => match[prop] !== undefined) 
                    ? match[possibleScoreProps.find(prop => match[prop] !== undefined)] 
                    : 'Not found';
                    
                // Try different property names that might contain the ID
                const possibleIdProps = ['_id', 'id', 'documentId'];
                const idValue = possibleIdProps.find(prop => match[prop] !== undefined)
                    ? match[possibleIdProps.find(prop => match[prop] !== undefined)]
                    : 'Not found';
                    
                console.log(`  ID (tried multiple properties): ${idValue}`);
                console.log(`  Score (tried multiple properties): ${scoreValue}`);
                console.log(`  Metadata:`, match.metadata);
                console.log(`  Content: ${match.metadata?.chunk || "No chunk content"}`);
                console.log("-------------------");
            });
            console.log("=== END OF CHUNKS ===");
            
            const concatenatedRetrievals = queryResponse.matches
                .map((match, index) => `\nClinical Finding ${index+1}: \n ${match.metadata?.chunk}`)
                .join(". \n\n");
            return concatenatedRetrievals;
        }
        return "<nomatches>";
    } catch (error) {
        console.error('Error in vector store query:', error);
        return `<error: ${error.message}>`;
    }
}