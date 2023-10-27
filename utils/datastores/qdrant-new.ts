import axios, { AxiosError, AxiosInstance } from "axios";
import { OpenAIEmbeddings } from "langchain/embeddings/openai";
import { z } from "zod";
import { QdrantVectorStore } from "langchain/vectorstores/qdrant";
import { QdrantClient } from "@qdrant/js-client-rest";
import { Chunk, MetadataFields } from "@/types";

import uuidv4 from "../uuid";

type Point = {
  id: string;
  vector: number[];
  payload: Omit<Chunk["metadata"], "chunk_id"> & {
    content: string;
  };
};

export class QdrantManager {
  // client: AxiosInstance;
  embeddings: OpenAIEmbeddings;
  quadrantVectorStore: QdrantVectorStore;
  collectionName: string = "text-embedding-ada-004";

  constructor() {
    this.embeddings = new OpenAIEmbeddings();

    this.quadrantVectorStore = new QdrantVectorStore(this.embeddings, {
      url: process.env.QDRANT_API_URL,
      // apiKey: process.env.QDRANT_API_KEY,
      collectionName: this.collectionName,
    });
  }

  private async initAppCollection() {
    await this.quadrantVectorStore.client.recreateCollection(
      "text-embedding-ada-004",
      {
        hnsw_config: {
          payload_m: 16,
          m: 0,
        },
        optimizers_config: {
          memmap_threshold: 10000,
        },
        vectors: {
          size: 1536,
          distance: "Cosine",
        },
        on_disk_payload: true,
      }
    );

    await this.quadrantVectorStore.client.createPayloadIndex(
      "text-embedding-ada-004",
      {
        field_name: MetadataFields.datastore_id,
        field_schema: "keyword",
      }
    );

    await this.quadrantVectorStore.client.createPayloadIndex(
      "text-embedding-ada-004",
      {
        field_name: MetadataFields.datasource_id,
        field_schema: "keyword",
      }
    );

    await this.quadrantVectorStore.client.createPayloadIndex(
      "text-embedding-ada-004",
      {
        field_name: MetadataFields.tags,
        field_schema: "keyword",
      }
    );

    await this.quadrantVectorStore.client.createPayloadIndex(
      "text-embedding-ada-004",
      {
        field_name: MetadataFields.custom_id,
        field_schema: "keyword",
      }
    );
  }

  private async addDocuments(
    documents: Chunk[],
    ids?: string[]
  ): Promise<void> {
    const texts = documents.map(({ pageContent }) => pageContent);
    return this.addVectors(
      await this.embeddings.embedDocuments(texts),
      documents,
      ids
    );
  }

  private async addVectors(
    vectors: number[][],
    documents: Chunk[],
    ids?: string[]
  ): Promise<void> {
    const documentIds = ids == null ? documents.map(() => uuidv4()) : ids;
    const qdrantVectors = vectors.map(
      (vector, idx) =>
        ({
          id: documentIds[idx],
          payload: {
            datastore_id: process.env.DATASTORE_ID as string,
            content: documents[idx].pageContent,
            source: documents[idx].metadata.source,
            tags: documents[idx].metadata.tags,
            chunk_hash: documents[idx].metadata.chunk_hash,
            chunk_offset: documents[idx].metadata.chunk_offset,
            datasource_hash: documents[idx].metadata.datasource_hash,
            datasource_id: documents[idx].metadata.datasource_id,
            custom_id: documents[idx].metadata.custom_id,
          },
          vector,
        } as Point)
    );

    await this.quadrantVectorStore.client.upsert("text-embedding-ada-004", {
      points: qdrantVectors,
    });
  }

  //  Delete points related to a Datastore
  async delete() {
    return this.quadrantVectorStore.client.delete("text-embedding-ada-004", {
      filter: {
        must: [
          {
            key: MetadataFields.datastore_id,
            match: { value: process.env.DATASTORE_ID as string },
          },
        ],
      },
    });
  }

  //  Delete points related to a Datasource
  async remove(datasourceId: string) {
    return this.quadrantVectorStore.client.delete("text-embedding-ada-004", {
      filter: {
        must: [
          {
            key: MetadataFields.datasource_id,
            match: {
              value: datasourceId,
            },
          },
        ],
      },
    });
  }

  async upload(documents: Chunk[]) {
    const ids: string[] = documents.map(
      (each) => each.metadata.chunk_id
    ) as string[];
    const datasourceId = documents[0].metadata.datasource_id as string;
    await this.initAppCollection();
    try {
      await this.remove(datasourceId);
      await this.addDocuments(documents, ids);
    } catch (error) {
      console.log("error is ", error);
      // if (axios.isAxiosError(error)) {
      //   if ((error as AxiosError).response?.status === 404) {
      //     // Collection does not exist, create it
      //     await this.initAppCollection();
      //     await this.addDocuments(documents, ids);
      //   }
      // } else {
      //   console.log(error);
      //   throw error;
      // }
    }

    return documents;
  }

  async search(props: any) {
    const vectors = await this.embeddings.embedDocuments([props.query]);

    const results = await this.quadrantVectorStore.client.search(
      "text-embedding-ada-004",
      {
        vector: vectors[0],
        limit: props.topK || 4,
        with_payload: true,
        with_vector: false,
        filter: {
          must: [
            {
              key: MetadataFields.datastore_id,
              match: { value: process.env.DATASTORE_ID as string },
            },
            ...(props.filters?.custom_id
              ? [
                  {
                    key: MetadataFields.custom_id,
                    match: { value: props.filters.custom_id },
                  },
                ]
              : []),
            ...(props.filters?.datasource_id
              ? [
                  {
                    key: MetadataFields.datasource_id,
                    match: { value: props.filters.datasource_id },
                  },
                ]
              : []),
          ],
        },
      }
    );

    return (results || [])?.map((each: any) => ({
      score: each?.score,
      source: each?.payload?.source,
      content: each?.payload?.text,
    }));
  }
}
