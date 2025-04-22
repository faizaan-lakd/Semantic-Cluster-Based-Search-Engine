import org.apache.commons.math3.ml.clustering.Clusterable;
import org.apache.commons.math3.ml.clustering.KMeansPlusPlusClusterer;
import org.apache.commons.math3.ml.clustering.Cluster;
import org.apache.lucene.analysis.standard.StandardAnalyzer;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.Field;
import org.apache.lucene.document.StringField;
import org.apache.lucene.document.TextField;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.IndexReader;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.queryparser.classic.QueryParser;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.Query;
import org.apache.lucene.search.ScoreDoc;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.RAMDirectory;
import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer;
import org.deeplearning4j.models.word2vec.Word2Vec;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;


import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.util.*;
import java.util.stream.Collectors;

public class SemanticClusterSearchEngine {
    private Directory index;
    private StandardAnalyzer analyzer;
    private Word2Vec word2VecModel;
    private Map<String, Paper> paperIndex;

    // Paper representation class
    public static class Paper {
        String id;
        String title;
        String authors;
        String year;
        String venue;
        String abstractPaper;
        List<String> references;
        double[] semanticVector;
        double pageRankScore;

        // Getters and toString for debugging
        @Override
        public String toString() {
            return String.format("Paper{title='%s', year='%s', venue='%s'}", 
                title, year, venue);
        }
    }

    // Wrapper class to make Paper clusterable
    private static class ClusterablePaper implements Clusterable {
        private final Paper paper;
        private final double[] vector;

        public ClusterablePaper(Paper paper, double[] vector) {
            this.paper = paper;
            this.vector = vector;
        }

        @Override
        public double[] getPoint() {
            return vector;
        }

        public Paper getPaper() {
            return paper;
        }
    }

    public SemanticClusterSearchEngine(String datasetPath, String word2vecModelPath) throws Exception {
        // Initialize Lucene components
        analyzer = new StandardAnalyzer();
        index = new RAMDirectory();
        paperIndex = new HashMap<>();

        // Load Word2Vec model
        System.out.println("Loading Word2Vec model...");
        word2VecModel = WordVectorSerializer.readWord2VecModel(new File(word2vecModelPath));
        System.out.println("Word2Vec model loaded.");
        // Load and index papers
        loadPapers(datasetPath);
        createLuceneIndex();
        computePageRankScores();
    }

    private void loadPapers(String datasetPath) throws Exception {
        try (BufferedReader br = new BufferedReader(new FileReader(datasetPath))) {
            Paper currentPaper = null;
            String line;
            while ((line = br.readLine()) != null) {
                if (line.startsWith("#*")) {
                    if (currentPaper != null) {
                        paperIndex.put(currentPaper.id, currentPaper);
                    }
                    currentPaper = new Paper();
                    currentPaper.title = line.substring(2).trim();
                } else if (line.startsWith("#@")) {
                    currentPaper.authors = line.substring(2).trim();
                } else if (line.startsWith("#t")) {
                    currentPaper.year = line.substring(2).trim();
                } else if (line.startsWith("#c")) {
                    currentPaper.venue = line.substring(2).trim();
                } else if (line.startsWith("#index")) {
                    currentPaper.id = line.substring(6).trim();
                } else if (line.startsWith("#%")) {
                    if (currentPaper.references == null) {
                        currentPaper.references = new ArrayList<>();
                    }
                    currentPaper.references.add(line.substring(2).trim());
                } else if (line.startsWith("#!")) {
                    currentPaper.abstractPaper = line.substring(2).trim();
                    // Compute semantic vector
                    currentPaper.semanticVector = computeSemanticVector(currentPaper.title);
                }
            }
            // Add last paper
            if (currentPaper != null) {
                paperIndex.put(currentPaper.id, currentPaper);
            }
        }
    }

    private double[] computeSemanticVector(String text) {
        String[] words = text.toLowerCase().split("\\s+");
        List<INDArray> wordVectors = new ArrayList<>();

        for (String word : words) {
            if (word2VecModel.hasWord(word)) {
                wordVectors.add(word2VecModel.getWordVectorMatrix(word));
            }
        }

        if (wordVectors.isEmpty()) {
            return new double[word2VecModel.getLayerSize()];
        }

        // Average word vectors
        INDArray documentVector = Nd4j.mean(Nd4j.vstack(wordVectors), 0);
        return documentVector.toDoubleVector();
    }

    private void createLuceneIndex() throws Exception {
        System.out.println("Creating Lucene index...");
        int paperCount = paperIndex.size();
        int processedCount = 0;

        IndexWriterConfig config = new IndexWriterConfig(analyzer);
        try (IndexWriter w = new IndexWriter(index, config)) {
            for (Paper paper : paperIndex.values()) {
                Document doc = new Document();

                // Add fields with null checks
                if (paper.id != null) {
                    doc.add(new StringField("id", paper.id, Field.Store.YES));
                }
                if (paper.title != null) {
                    doc.add(new TextField("title", paper.title, Field.Store.YES));
                } else {
                    doc.add(new TextField("title", "", Field.Store.YES)); // Default empty title if null
                }
                if (paper.abstractPaper != null) {
                    doc.add(new TextField("abstractPaper", paper.abstractPaper, Field.Store.YES));
                } else {
                    doc.add(new TextField("abstractPaper", "", Field.Store.YES)); // Default empty abstract if null
                }
                if (paper.authors != null) {
                    doc.add(new StringField("authors", paper.authors, Field.Store.YES));
                } else {
                    doc.add(new StringField("authors", "", Field.Store.YES)); // Default empty authors if null
                }
                if (paper.year != null) {
                    doc.add(new StringField("year", paper.year, Field.Store.YES));
                } else {
                    doc.add(new StringField("year", "", Field.Store.YES)); // Default empty year if null
                }
                if (paper.venue != null) {
                    doc.add(new StringField("venue", paper.venue, Field.Store.YES));
                } else {
                    doc.add(new StringField("venue", "", Field.Store.YES)); // Default empty venue if null
                }

                w.addDocument(doc);

                processedCount++;
                if (processedCount % 10 == 0) {
                    int progress = (int) ((double) processedCount / paperCount * 100);
                    System.out.println("Progress creating Lucene index: " + progress + "%");
                }
            }
        }
        System.out.println("Lucene index created with " + paperCount + " documents.");
    }

    private void computePageRankScores() {
    	int totalPapers = paperIndex.size();
        int processedPapers = 0;
    	
        // Simplified PageRank computation
        Map<String, Set<String>> citationGraph = new HashMap<>();
        
        // Build citation graph
        for (Paper paper : paperIndex.values()) {
            if (paper.references != null) {
                citationGraph.put(paper.id, new HashSet<>(paper.references));
            }
        }

        // Basic iterative PageRank
        Map<String, Double> pageRankScores = new HashMap<>();
        double dampingFactor = 0.85;
        int iterations = 1;
        
        // Initialize uniform distribution
        for (String paperId : paperIndex.keySet()) {
            pageRankScores.put(paperId, 1.0 / paperIndex.size());
        }

        for (int i = 0; i < iterations; i++) {
            Map<String, Double> newScores = new HashMap<>();
            
            for (String paperId : paperIndex.keySet()) {
                double score = (1 - dampingFactor) / paperIndex.size();
                
                for (String otherPaperId : paperIndex.keySet()) {
                    if (citationGraph.getOrDefault(otherPaperId, Collections.emptySet()).contains(paperId)) {
                        score += dampingFactor * (pageRankScores.get(otherPaperId) / 
                                citationGraph.get(otherPaperId).size());
                    }
                }
                
                newScores.put(paperId, score);
            }
            
            pageRankScores = newScores;
            processedPapers++;
            if (processedPapers % 10 == 0) {
                int progress = (int) ((double) processedPapers / totalPapers * 100);
                System.out.println("Progress computing PageRank: " + progress + "%");
            }
        }

        // Update paper PageRank scores
        for (Paper paper : paperIndex.values()) {
            paper.pageRankScore = pageRankScores.get(paper.id)*100000;
        }
        System.out.println("PageRank computation completed.");
    }

    public List<List<Paper>> semanticSearchWithClustering(String queryText, int topN, int numClusters) throws Exception {
        System.out.println("Starting semantic search with clustering...");
        
        // Convert query to semantic vector
        double[] queryVector = computeSemanticVector(queryText);
        System.out.println("Query converted to semantic vector.");

        // Perform Lucene text search
        System.out.println("Performing Lucene text search...");
        IndexReader reader = DirectoryReader.open(index);
        IndexSearcher searcher = new IndexSearcher(reader);
        QueryParser parser = new QueryParser("title", analyzer);
        Query query = parser.parse(queryText);
        
        TopDocs results = searcher.search(query, topN * 10);  // More results for clustering
        
        // Create a list to store papers with their combined scores
        List<PaperScore> paperScores = new ArrayList<>();
        
        for (ScoreDoc scoreDoc : results.scoreDocs) {
            Document doc = searcher.doc(scoreDoc.doc);
            Paper paper = paperIndex.get(doc.get("id"));
            
            // Check for null or empty semantic vector
            if (paper.semanticVector == null || paper.semanticVector.length == 0) {
                continue;  // Skip papers with no valid semantic vector
            }
            
            // Compute semantic similarity
            INDArray paperVec = Nd4j.create(paper.semanticVector);
            INDArray queryVec = Nd4j.create(queryVector);
            double semanticScore = Transforms.cosineSim(paperVec, queryVec);
            
            // Combine scores with weighted approach
            // Lucene text search score (0.5), semantic similarity (0.3), PageRank (0.2)
            double combinedScore = (0.5 * scoreDoc.score) + 
                                   (0.2 * semanticScore) + 
                                   (0.3 * (paper.pageRankScore / 100000.0));
            
            paperScores.add(new PaperScore(paper, combinedScore));
        }

        // Sort papers by combined score in descending order
        paperScores.sort(Comparator.comparingDouble(ps -> -ps.score));

        // Select top N papers for clustering
        List<ClusterablePaper> clusterablePapers = new ArrayList<>();
        for (int i = 0; i < Math.min(topN, paperScores.size()); i++) {
            PaperScore paperScore = paperScores.get(i);
            
            // Create enhanced vector with combined score
            double[] enhancedVector = Arrays.copyOf(paperScore.paper.semanticVector, 
                                                    paperScore.paper.semanticVector.length + 1);
            enhancedVector[enhancedVector.length - 1] = paperScore.score;
            
            clusterablePapers.add(new ClusterablePaper(paperScore.paper, enhancedVector));
        }

        // Perform K-means clustering
        System.out.println("performing k-means...");
        KMeansPlusPlusClusterer<ClusterablePaper> clusterer = 
            new KMeansPlusPlusClusterer<>(numClusters, 100);
        
        // Convert CentroidCluster to our cluster representation
        List<List<Paper>> clusteredResults = new ArrayList<>();
        List<org.apache.commons.math3.ml.clustering.CentroidCluster<ClusterablePaper>> centroidClusters = 
            (List<org.apache.commons.math3.ml.clustering.CentroidCluster<ClusterablePaper>>) clusterer.cluster(clusterablePapers);

        for (org.apache.commons.math3.ml.clustering.CentroidCluster<ClusterablePaper> centroidCluster : centroidClusters) {
            List<Paper> clusterPapers = centroidCluster.getPoints().stream()
                .map(ClusterablePaper::getPaper)
                .collect(Collectors.toList());
            
            clusteredResults.add(clusterPapers);
        }
        System.out.println("clustering done");

        return clusteredResults;
    }

    // New helper class to track paper and its combined score
    private static class PaperScore {
        Paper paper;
        double score;

        public PaperScore(Paper paper, double score) {
            this.paper = paper;
            this.score = score;
        }
    }

    // Method to print clustered results
    private static void printClusteredResults(List<List<Paper>> clusters) {
        System.out.println("===== Clustered Search Results =====");
        
        for (int i = 0; i < clusters.size(); i++) {
            System.out.println("\n--- Cluster #" + (i+1) + " ---");
            List<Paper> clusterPapers = clusters.get(i);
            
            for (int j = 0; j < clusterPapers.size(); j++) {
                Paper paper = clusterPapers.get(j);
                System.out.println("\n[Cluster " + (i+1) + " - Paper #" + (j+1) + "]");
                System.out.println("Title: " + paper.title);
                System.out.println("Authors: " + paper.authors);
                System.out.println("Year: " + paper.year);
                System.out.println("Venue: " + paper.venue);
                
                // Truncate abstractPaper
                String abstractPaperText = paper.abstractPaper;
                if (abstractPaperText.length() > 200) {
                    abstractPaperText = abstractPaperText.substring(0, 200) + "...";
                }
                System.out.println("abstractPaper: " + abstractPaperText);
                System.out.printf("PageRank Score: " + paper.pageRankScore);
            }
        }
        
        System.out.println("\nTotal Clusters: " + clusters.size());
    }

    public static void main(String[] args) {
        try {
            // Paths to dataset and Word2Vec model (you'll need to replace these with actual paths)
            String datasetPath = "C:\\Users\\faiza\\Downloads\\citation-network1\\outputacmTrimmed.txt";
            String word2vecModelPath = "C:\\Users\\faiza\\Downloads\\citation-network1\\GoogleNews-vectors-negative300.bin";

            // Initialize the Semantic Cluster Search Engine
            SemanticClusterSearchEngine searchEngine = 
                new SemanticClusterSearchEngine(datasetPath, word2vecModelPath);

            // Example search query
            String searchQuery = "Neural Networks";

            // Perform semantic search with clustering
            // Parameters: 
            // - Search query
            // - Top N results to retrieve
            // - Number of clusters to create
            List<List<Paper>> clusteredResults = searchEngine.semanticSearchWithClustering(
                searchQuery, 
                20,  // Retrieve top 20 results 
                5   // Create 5 clusters
            );

            // Print the clustered results
            printClusteredResults(clusteredResults);

        } catch (Exception e) {
            System.err.println("Error occurred during semantic search:");
            e.printStackTrace();
        }
    }
}