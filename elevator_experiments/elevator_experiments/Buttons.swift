//
//  Buttons.swift
//  elevator_experiments
//
//  Created by Pankaj Sharma on 1/27/23.
//

import SwiftUI
import Charts

struct WeekdayData: Identifiable {
    let id = UUID()
    var name: String
    var weekday :Int
    var prob :Float
    
    init(name: String, weekday :Int, prob :Float){
        self.weekday = weekday
        self.name = name
        self.prob = prob
    }
}

struct Buttons: View {
    @State var elProb :[String] = ["0.00","0.00","0.00"]
    @State var weekdata :[WeekdayData] = []
    /*
    [
        WeekdayData(name: "Elevator 1",weekday: 0,prob: 0.1),
        WeekdayData(name: "Elevator 1",weekday: 1,prob: 0.1),        WeekdayData(name: "Elevator 1",weekday: 2,prob: 0.1),        WeekdayData(name: "Elevator 1",weekday: 3,prob: 0.1),        WeekdayData(name: "Elevator 1",weekday: 4,prob: 0.1),        WeekdayData(name: "Elevator 1",weekday: 5,prob: 0.1),        WeekdayData(name: "Elevator 1",weekday: 6,prob: 0.1),
        WeekdayData(name: "Elevator 2",weekday: 0,prob: 0.2),
        WeekdayData(name: "Elevator 2",weekday: 1,prob: 0.2),        WeekdayData(name: "Elevator 2",weekday: 2,prob: 0.2),        WeekdayData(name: "Elevator 2",weekday: 3,prob: 0.2),        WeekdayData(name: "Elevator 2",weekday: 4,prob: 0.2),        WeekdayData(name: "Elevator 2",weekday: 5,prob: 0.2),        WeekdayData(name: "Elevator 2",weekday: 6,prob: 0.2),
        WeekdayData(name: "Elevator 3",weekday: 0,prob: 0.3),
        WeekdayData(name: "Elevator 3",weekday: 1,prob: 0.3),        WeekdayData(name: "Elevator 3",weekday: 2,prob: 0.3),        WeekdayData(name: "Elevator 3",weekday: 3,prob: 0.3),        WeekdayData(name: "Elevator 3",weekday: 4,prob: 0.3),        WeekdayData(name: "Elevator 3",weekday: 5,prob: 0.3),        WeekdayData(name: "Elevator 3",weekday: 6,prob: 0.3)

    ]
    */
    
    
    private var excelFile = "elevator_experiments.xlsx"
    private var tableName = "Table1"
    private func updateWithHelper(){
        self.elProb[0]=helper.el1Prob
        self.elProb[1]=helper.el2Prob
        self.elProb[2]=helper.el3Prob
        self.weekdata=helper.weekdayData
    }

    var body: some View {
            VStack{
                Spacer()
                HStack{
                    VStack{
                        HStack{
                            Button("Elevator 1") {
                                //helper.addData(input: "1")
                                Task{
                                    await helper.processTableRow(excelFile: self.excelFile, tableName: self.tableName, elevatorNo: 1)
                                    updateWithHelper()
                                        
                                }
                            }.padding()
                            Text(self.elProb[0])
                            Button("P"){}
                            Text("0.20")
                        }
                        HStack{
                            Button("Elevator 2") {
                                //helper.addData(input: "2")
                                Task{
                                    await helper.processTableRow(excelFile: self.excelFile, tableName: self.tableName, elevatorNo: 2)
                                    updateWithHelper()
                                }
                            }.padding()
                            Text(self.elProb[1])
                            Button("P"){}
                            Text("0.80")
                        }
                        HStack{
                            Button("Elevator 3") {
                                //helper.addData(input: "3")
                                Task{
                                    await helper.processTableRow(excelFile: self.excelFile, tableName: self.tableName, elevatorNo: 3)
                                    updateWithHelper()
                               }
                            }.padding()
                            Text(self.elProb[2])
                            Button("P"){}
                            Text("0.10")
                        }
                    }
                }.padding()
                /*Spacer()
                HStack{
                    Chart (content: {
                        BarMark(
                            x: .value("Elevator", 1),
                            y: .value("Probability", Float(self.elProb[0])!  )
                        )
                        BarMark(
                            x: .value("Elevator", 2),
                            y: .value("Probability", Float(self.elProb[1])!  )
                        )
                        BarMark(
                            x: .value("Elevator", 3),
                            y: .value("Probability", Float(self.elProb[2])!)
                        )
                    })

                }.frame(width: 250, height: 100).padding()*/
                Spacer()
                HStack{
                    Chart(self.weekdata) { item in
                        LineMark(
                            x: .value("Day", item.weekday),
                            y: .value("Prob", item.prob)
                        ).foregroundStyle(by: .value("Elevator", item.name))
                        
                    }.chartYScale(domain: 0.1...0.5)
                        .chartXScale(domain: 1...7)
                    
                    
                }.frame(width: 250, height: 200).padding()

                MyCustomLogin(value: self.$elProb, data: self.$weekdata).onOpenURL { url in
                    helper.openUrl(url: url)
                }

            }.padding()

        
    }
}

struct Buttons_Previews: PreviewProvider {
    static var previews: some View {
        Buttons()
    }
}

