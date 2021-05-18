const app = document.getElementById("root");

const logo = document.createElement('img');
logo.src = 'logo.png';

const container = document.createElement('div');
container.setAttribute('class','container');

app.appendChild(logo);
app.appendChild(container);


var request = new XMLHttpRequest();

request.open('GET','https://ghibliapi.herokuapp.com/films',true);

request.onload = function(){
	
	//Begin accessing json data
	var data = JSON.parse(this.response);
	
	if(request.status >= 200  && request.status < 400){
		data.forEach((movie) => {
			//console.log(movie.title);
			const card = document.createElement('div');
			card.setAttribute('class','card');

			const h1 = document.createElement('h1');
			h1.textContent = movie.title;

			const p = document.createElement('p');
			movie.description = movie.description.substring(0,300);
			p.textContent = `${movie.description}...`

			container.appendChild(card);
			container.appendChild(h1);
			container.appendChild(p);


		})	
}else{
	const errorMessage = document.createElement('loool');
	errorMessage.textContent = 'Dame da ne dame yo';
	app.appendChild(errorMessage);
}
}

request.send();


 