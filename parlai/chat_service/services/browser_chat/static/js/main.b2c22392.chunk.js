(this["webpackJsonpchatbot-webpage"]=this["webpackJsonpchatbot-webpage"]||[]).push([[0],{13:function(e,t,a){},14:function(e,t,a){},21:function(e,t,a){"use strict";a.r(t);var n=a(1),s=a.n(n),c=a(6),i=a.n(c),o=(a(13),a(14),a(3)),r=a.n(o),h={botName:"BlenderBot",initialMessages:[Object(o.createChatBotMessage)("Hi! Talk to me :D")]},u=a(8),g=a(5),b=a(4),d=function e(t,a,n){var s=this;Object(b.a)(this,e),this.greet=function(){var e=s.createChatBotMessage("Sorry, I am not connected to the backend yet :(");s.addMessageToState(e)},this.addMessageToState=function(e){s.setState((function(t){return Object(g.a)(Object(g.a)({},t),{},{messages:[].concat(Object(u.a)(t.messages),[e])})}))},this.createChatBotMessage=t,this.setState=a,this.createClientMessage=n},j=a(7),f=function(){function e(t,a){Object(b.a)(this,e),this.actionProvider=t,this.state=a}return Object(j.a)(e,[{key:"parse",value:function(e){console.log("user input: "+e),this.actionProvider.greet()}}]),e}(),l=a(0);var p=function(){return Object(l.jsx)("div",{className:"App",children:Object(l.jsx)("header",{className:"App-header",children:Object(l.jsx)(r.a,{config:h,actionProvider:d,messageParser:f})})})},O=function(e){e&&e instanceof Function&&a.e(3).then(a.bind(null,22)).then((function(t){var a=t.getCLS,n=t.getFID,s=t.getFCP,c=t.getLCP,i=t.getTTFB;a(e),n(e),s(e),c(e),i(e)}))};i.a.render(Object(l.jsx)(s.a.StrictMode,{children:Object(l.jsx)(p,{})}),document.getElementById("root")),O()}},[[21,1,2]]]);
//# sourceMappingURL=main.b2c22392.chunk.js.map