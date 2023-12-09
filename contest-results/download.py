from bs4 import BeautifulSoup
import requests
import os
import concurrent.futures
def main():
    __dir = os.path.dirname(os.path.abspath(__file__))
    # Load the HTML file
    file_path = os.path.join(__dir, 'pacman.html')

    with open(file_path, 'r') as file:
        content = file.read()

    # Parse the HTML content
    soup = BeautifulSoup(content, 'html.parser')

    # Define the team name
    team_name = "A* is born*"

    # Search for all rows in the table
    rows = soup.find_all('tr')

    # Function to determine the directory based on the game result
    def determine_directory(team_name, winner):
        if winner == team_name:
            return 'win'
        elif winner == 'None':
            return 'tie'
        else:
            return 'lose'

    # Base URL for file downloads
    base_url = "https://pacman-contest.upf.edu/final_2023/"  # Replace with the actual base URL

    # Prepare to store the file URLs and their respective directories
    file_urls = []

    # Process each row
    for row in rows:
        columns = row.find_all('td')
        if len(columns) >= 9:  # Ensure there are at least 9 columns
            team1, team2, _, _, _, winner, *file_links = columns[:9]  # Extract first 9 columns
            if team_name in [team1.get_text().strip(), team2.get_text().strip()]:
                dir_name = determine_directory(team_name, winner.get_text().strip())
                for link in file_links:
                    file_url = base_url + link.find('a')['href']
                    file_urls.append((file_url, dir_name))

    # Using ThreadPoolExecutor to download files in parallel
    with concurrent.futures.ThreadPoolExecutor() as executor:
        executor.map(download_file, file_urls)



# Function to download and save a file
def download_file(url_and_directory):
    url, directory = url_and_directory
    directory = os.path.join(os.path.dirname(os.path.abspath(__file__)), directory)
    # Extract the file name from the URL
    file_name = url.split('/')[-1]
    
    # Ensure the directory exists
    if not os.path.exists(directory):
        os.makedirs(directory)

    # Full path for the file to be saved
    file_path = os.path.join(directory, file_name)

    # Download and save the file
    response = requests.get(url)
    with open(file_path, 'wb') as file:
        file.write(response.content)

# Main script
if __name__ == "__main__":
    main()
