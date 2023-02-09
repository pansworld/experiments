//
//  Buttons.swift
//  elevator_experiments
//
//  Created by Pankaj Sharma on 1/27/23.
//

import SwiftUI

struct Buttons: View {
    @State var el1Prob :String = "0.00"
    @State var el2Prob :String = "0.00"
    @State var el3Prob :String = "0.00"


    var body: some View {
        VStack{
            HStack{
                Button("Elevator 1") {
                    helper.addData(input: "1")
                }.padding()
                Text(self.el1Prob)
            }
            HStack{
                Button("Elevator 2") {
                    helper.addData(input: "2")
                }.padding()
                Text(self.el2Prob)
            }
            HStack{
                Button("Elevator 3") {
                    helper.addData(input: "3")
                }.padding()
                Text(self.el2Prob)
            }
            Spacer()
            HStack{
                MyCustomLogin(value1: self.$el1Prob, value2: self.$el2Prob, value3: self.$el3Prob)
            }.padding()


        }.padding()
            
    }
}

struct Buttons_Previews: PreviewProvider {
    static var previews: some View {
        Buttons()
    }
}

