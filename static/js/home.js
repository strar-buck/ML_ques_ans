$(function(){
	$('#search-form').submit(function(event){
		event.preventDefault();
		// location.reload()
		var form_data=$(this).serializeArray();
		form_data = JSON.stringify(form_data);
		
		console.log(form_data);
		// var back = ["grey", "orange", "yellow", "black", "green","pink"];
  // 		var rand = back[Math.floor(Math.random() * back.length)];		
		//var question=$('#search').val()
		//alert(question)
		$.ajax({
	           	url:'/predict_tag/',
	           	type:'post',
	            data:{form_data:form_data},
	            success: function(data){
	            	console.log(data)
	            	$('#search_tag').empty();
	            	// window.location.reload(); 
	            	// alert(data.length)
	  				for(i=0;i<data.length;i++){
	  					// $('#search_tag').css('display','inline-block')
	  				//	var color=get_random_color();
	  					var html_div="<div class='search_tag'>";
	  				 		html_div += "<div class='tag'>"  + data[i] + "</div>";
	  				 	
	                	
	                    $('#search_tag').append(html_div);
	                    $('.search_tag').css('color','black').css('background',get_random_color());
	                    
	                	html_div+= "</div>";

	                    
	                }    //alert("After adding form_data"); 


	        	}
	       	});

		});	
	});	

// getting random color for tags

function get_random_color() {
  // function c() {
   var letters='0123456789ABCDEF';
   var color="#";

   for(var i=0;i<6;i++){
   		color+=letters[Math.floor(Math.random() * 16 )];

   }
   return color;
}