// MongoDB initialization script for production
// Creates application user with appropriate permissions

// Switch to admin database
db = db.getSiblingDB('admin');

// Create application database
db = db.getSiblingDB('rag_production');

// Create application user
db.createUser({
  user: 'rag_user',
  pwd: 'rag_secure_password_change_me',
  roles: [
    {
      role: 'readWrite',
      db: 'rag_production'
    },
    {
      role: 'dbAdmin',
      db: 'rag_production'
    }
  ]
});

// Create collections with validation
db.createCollection('documents', {
  validator: {
    $jsonSchema: {
      bsonType: 'object',
      required: ['file_path', 'content', 'metadata', 'created_at'],
      properties: {
        file_path: {
          bsonType: 'string',
          description: 'Path to the source document file'
        },
        content: {
          bsonType: 'string',
          description: 'Document content'
        },
        metadata: {
          bsonType: 'object',
          description: 'Document metadata'
        },
        created_at: {
          bsonType: 'date',
          description: 'Document creation timestamp'
        }
      }
    }
  }
});

db.createCollection('chunks', {
  validator: {
    $jsonSchema: {
      bsonType: 'object',
      required: ['document_id', 'chunk_text', 'chunk_index', 'metadata'],
      properties: {
        document_id: {
          bsonType: 'objectId',
          description: 'Reference to parent document'
        },
        chunk_text: {
          bsonType: 'string',
          description: 'Chunk content'
        },
        chunk_index: {
          bsonType: 'int',
          description: 'Chunk position in document'
        },
        metadata: {
          bsonType: 'object',
          description: 'Chunk metadata'
        }
      }
    }
  }
});

db.createCollection('queries', {
  validator: {
    $jsonSchema: {
      bsonType: 'object',
      required: ['query_text', 'timestamp', 'response_time'],
      properties: {
        query_text: {
          bsonType: 'string',
          description: 'User query text'
        },
        timestamp: {
          bsonType: 'date',
          description: 'Query timestamp'
        },
        response_time: {
          bsonType: 'double',
          description: 'Query response time in seconds'
        }
      }
    }
  }
});

// Create indexes for performance
db.documents.createIndex({ "file_path": 1 }, { unique: true });
db.documents.createIndex({ "created_at": 1 });
db.documents.createIndex({ "metadata.source": 1 });

db.chunks.createIndex({ "document_id": 1 });
db.chunks.createIndex({ "chunk_index": 1 });
db.chunks.createIndex({ "document_id": 1, "chunk_index": 1 }, { unique: true });

db.queries.createIndex({ "timestamp": 1 });
db.queries.createIndex({ "query_text": "text" });

print('✅ Production database initialized successfully');
print('📊 Created collections: documents, chunks, queries');
print('🔐 Created user: rag_user');
print('⚡ Created performance indexes');
print('⚠️  Remember to change the default password!');