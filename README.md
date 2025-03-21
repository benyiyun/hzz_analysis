1. Once all files are downloaded, run the command 'docker-compose build --no-cache'
2. To start the system, run 'docker-compose up -d'
3. To copy the output plot into the directory where the container is built, run 'docker cp plotter_container:/app/shared/output_plot.png ./output_plot.png'
4. Container logs can be viewed by running 'docker logs <container_name>'
