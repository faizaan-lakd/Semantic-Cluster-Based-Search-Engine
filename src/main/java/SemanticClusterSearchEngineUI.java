import javax.swing.*;
import javax.swing.table.DefaultTableModel;
import java.awt.*;
import java.util.List;

public class SemanticClusterSearchEngineUI {
    private SemanticClusterSearchEngine searchEngine;

    public SemanticClusterSearchEngineUI(SemanticClusterSearchEngine searchEngine) {
        this.searchEngine = searchEngine;
        initializeUI();
    }

    private void initializeUI() {
        JFrame frame = new JFrame("Semantic Cluster Search Engine");
        frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        frame.setSize(800, 600);

        // Top panel with search bar
        JPanel topPanel = new JPanel(new BorderLayout());
        JLabel searchLabel = new JLabel("Enter Search Query: ");
        JTextField searchField = new JTextField();
        JButton searchButton = new JButton("Search");

        topPanel.add(searchLabel, BorderLayout.WEST);
        topPanel.add(searchField, BorderLayout.CENTER);
        topPanel.add(searchButton, BorderLayout.EAST);

        // Table for displaying clustered results
        JTable resultTable = new JTable();
        DefaultTableModel tableModel = new DefaultTableModel(
            new Object[]{"Cluster", "Title", "Authors", "Year", "Venue", "PageRank Score"},
            0
        );
        resultTable.setModel(tableModel);
        JScrollPane tableScrollPane = new JScrollPane(resultTable);

        // Add components to frame
        frame.setLayout(new BorderLayout());
        frame.add(topPanel, BorderLayout.NORTH);
        frame.add(tableScrollPane, BorderLayout.CENTER);

        // Search button action listener
        searchButton.addActionListener(e -> {
            String queryText = searchField.getText().trim();
            if (queryText.isEmpty()) {
                JOptionPane.showMessageDialog(frame, "Please enter a search query.", "Error", JOptionPane.ERROR_MESSAGE);
                return;
            }

            // Perform semantic search with clustering
            try {
                List<List<SemanticClusterSearchEngine.Paper>> clusteredResults = searchEngine.semanticSearchWithClustering(queryText, 20, 5);

                // Populate table with results
                tableModel.setRowCount(0); // Clear previous results
                for (int clusterIndex = 0; clusterIndex < clusteredResults.size(); clusterIndex++) {
                    List<SemanticClusterSearchEngine.Paper> cluster = clusteredResults.get(clusterIndex);
                    for (SemanticClusterSearchEngine.Paper paper : cluster) {
                        tableModel.addRow(new Object[]{
                            clusterIndex + 1,
                            paper.title,
                            paper.authors,
                            paper.year,
                            paper.venue,
                            paper.pageRankScore
                        });
                    }
                }
            } catch (Exception ex) {
                JOptionPane.showMessageDialog(frame, "Error during search: " + ex.getMessage(), "Error", JOptionPane.ERROR_MESSAGE);
                ex.printStackTrace();
            }
        });

        frame.setVisible(true);
    }

    public static void main(String[] args) {
        try {
            // Paths to dataset and Word2Vec model (replace with actual paths)
            String datasetPath = "C:\\Users\\faiza\\Downloads\\citation-network1\\outputacmTrimmedShort.txt";
            String word2vecModelPath = "C:\\Users\\faiza\\Downloads\\citation-network1\\GoogleNews-vectors-negative300.bin";

            // Initialize the Semantic Cluster Search Engine
            SemanticClusterSearchEngine searchEngine = new SemanticClusterSearchEngine(datasetPath, word2vecModelPath);

            // Launch the UI
            SwingUtilities.invokeLater(() -> new SemanticClusterSearchEngineUI(searchEngine));
        } catch (Exception e) {
            System.err.println("Error initializing the search engine:");
            e.printStackTrace();
        }
    }
}
