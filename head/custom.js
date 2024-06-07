var tab = function(id) {
  var myTab = document.getElementById(id);    
  var myUl = myTab.getElementsByTagName("ul")[0];
  var myLi = myUl.getElementsByTagName("li");   
  var myDiv = myTab.getElementsByTagName("div"); 
  for(var i = 0; i < myLi.length; i++) {
      myLi[i].index = i;
      myLi[i].onmouseover = function() {
          if (this.className == "off") {
              this.className = "ho";
          }
      }
      myLi[i].onmouseout = function() {
          if (this.className == "ho") {
              this.className = "off";
          }
      }
      myLi[i].onclick = function() {
          for(var j = 0; j < myLi.length; j++) {
              myLi[j].className = "off";
              myDiv[j].className = "hide";
          }
          this.className = "on";
          myDiv[this.index].className = "show";
      }
  }
}
window.onload = function() {
  tab("tab1");
  tab("tab2");
  tab("tab3");
  tab("tab4");
  tab("tab5");
}
